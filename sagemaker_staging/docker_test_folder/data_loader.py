import os
import multiprocessing
import rasterio
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import tensorflow.keras as keras
import keras
import pandas as pd
import boto3
import io
import json
from tensorflow_addons.metrics import F1Score, HammingLoss
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import random
from save_checkpoint_callback_custom import SaveCheckpoints
import wandb
from wandb.keras import WandbCallback
from rasterio.session import AWSSession

class DataLoader:
    def __init__(self, label_file_path_train="labels_test_v1.csv",
                 label_file_path_val="labels_val.csv",
                 label_mapping_path="labels.json",
                 s3_file_paths=True,
                 bucket_name='canopy-production-ml',
                 data_extension_type='.tif',
                 bands=['all'],
                 training_data_shape=(100, 100, 18),
                 augment=False,
                 random_flip_up_down=True,
                 random_flip_left_right=True,
                 flip_left_right=True,
                 flip_up_down=True,
                 rot90=True,
                 transpose=False,
                 enable_shuffle=False,
                 training_data_shuffle_buffer_size=1000,
                 training_data_batch_size=20,
                 enable_data_prefetch=False,
                 data_prefetch_size=tf.data.experimental.AUTOTUNE,
                 num_parallel_calls=2 * multiprocessing.cpu_count(),
                 output_shape=(tf.float32, tf.float32)):

        self.s3 = boto3.resource('s3')
        self.bucket_name = bucket_name
        self.s3_file_paths = s3_file_paths

        if self.s3_file_paths:

            self.label_file_path_train = self.read_s3_obj(label_file_path_train)
            self.label_file_path_val = self.read_s3_obj(label_file_path_val)
            self.label_mapping_path = self.read_s3_obj(label_mapping_path)

        else:

            self.label_file_path_train = label_file_path_train
            self.label_file_path_val = label_file_path_val
            self.label_mapping_path = label_mapping_path

        self.labels_file_train = pd.read_csv(self.label_file_path_train)
        self.labels_file_val = pd.read_csv(self.label_file_path_val)
        self.training_filenames = self.labels_file_train.paths.to_list()

        self.labels_file_val = pd.read_csv(self.label_file_path_val)
        self.validation_filenames = self.labels_file_val.paths.to_list()

        self.bands = bands

        self.augment = augment
        self.random_flip_up_down = random_flip_up_down
        self.random_flip_left_right = random_flip_left_right
        self.flip_left_right = flip_left_right
        self.flip_up_down = flip_up_down
        self.rot90 = rot90
        self.transpose = transpose

        self.enable_shuffle = enable_shuffle
        if training_data_shuffle_buffer_size is not None:
            self.training_data_shuffle_buffer_size = training_data_shuffle_buffer_size

        self.num_parallel_calls = num_parallel_calls
        self.enable_data_prefetch = enable_data_prefetch
        self.data_prefetch_size = data_prefetch_size
        self.data_extension_type = data_extension_type
        self.output_shape = output_shape

        self.training_data_shape = training_data_shape

        self.training_data_batch_size = training_data_batch_size

        self.class_weight_train = self.generate_class_weight(self.label_file_path_train, self.labels_file_train)
        self.class_weight_val = self.generate_class_weight(self.label_file_path_val, self.labels_file_val)

        self.build_training_dataset()
        self.build_validation_dataset()

    def read_s3_obj(self, s3_key):
        s3 = self.s3
        obj = s3.Object(self.bucket_name, s3_key)
        obj_bytes = io.BytesIO(obj.get()['Body'].read())
        return obj_bytes

    def build_training_dataset(self):
        # Tensor of all the paths to the images
        self.training_dataset = tf.data.Dataset.from_tensor_slices(self.training_filenames)

        # If data augmentation
        if self.augment is True:
            print(f"Data augmentation enabled: flip_up_down {self.flip_up_down}, "
                  f"flip_left_right {self.flip_left_right}, rot90 {self.rot90}")
            datasets = []  # list of all datasets to be combined into the training dataset
            # data_augmentations=1 because even without data augmentation we need the dataset once (read more about repeat function for this)
            data_augmentations = 1

            # Original dataset. No augmentation performed here
            # tf.py_function: Wraps a python function into a TensorFlow op that executes it eagerly,
            # This function allows expressing computations in a TensorFlow graph as Python functions
            self.training_dataset_non_augmented = self.training_dataset.map((
                lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
                num_parallel_calls=self.num_parallel_calls)
            datasets.append(self.training_dataset_non_augmented)

            ### Below are the data augmentations that use the original images
            # Outputs the contents of `image` flipped along the width dimension
            if self.flip_left_right is True:
                self.training_dataset_augmented_flip_left_right = self.training_dataset.map((
                    lambda x: tf.py_function(self.process_path_train_set_augment_flip_left_right, [x],
                                             self.output_shape)),
                    num_parallel_calls=self.num_parallel_calls)
                datasets.append(self.training_dataset_augmented_flip_left_right)
                data_augmentations = data_augmentations + 1

            # Outputs the contents of `image` flipped along the height dimension
            if self.flip_up_down is True:
                self.training_dataset_augmented_flip_up_down = self.training_dataset.map((
                    lambda x: tf.py_function(self.process_path_train_set_augment_flip_up_down, [x], self.output_shape)),
                    num_parallel_calls=self.num_parallel_calls)
                datasets.append(self.training_dataset_augmented_flip_up_down)
                data_augmentations = data_augmentations + 1

            # Rotate image(s) counter-clockwise by 90 degrees
            if self.rot90 is True:
                self.training_dataset_augmented_rot90 = self.training_dataset.map((
                    lambda x: tf.py_function(self.process_path_train_set_augment_rot90, [x], self.output_shape)),
                    num_parallel_calls=self.num_parallel_calls)
                datasets.append(self.training_dataset_augmented_rot90)
                data_augmentations = data_augmentations + 1

            # Define a dataset containing `[0, 1, 0, 1, 0, 1, ..., 0, 1]` if data_augmentations = 1 for example
            # This line of code basically makes it possible to "repeat" the original dataset with augmentations
            choice_dataset = tf.data.Dataset.range(data_augmentations).repeat(len(self.training_filenames))

            # TODO to some performance testing on the choose_from_datasets duplication augmentation method done above
            self.training_dataset = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
            self.length_training_dataset = len(self.training_filenames) * len(datasets)
            print(f"Training on {self.length_training_dataset} images")
        else:
            print("No data augmentation. Please set augment to True if you want to augment training dataset")
            self.training_dataset = self.training_dataset.map((
                lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
                num_parallel_calls=self.num_parallel_calls)
            self.length_training_dataset = len(self.training_filenames)
            print(f"Training on {self.length_training_dataset} images")

        # Randomly shuffles the elements of this dataset.
        # This dataset fills a buffer with `buffer_size` elements, then randomly
        # samples elements from this buffer, replacing the selected elements with new
        # elements. For perfect shuffling, a buffer size greater than or equal to the
        # full size of the dataset is required.
        if self.enable_shuffle is True:
            if self.training_data_shuffle_buffer_size is None:
                self.training_data_shuffle_buffer_size = len(self.length_training_dataset)
            self.training_dataset = self.training_dataset.shuffle(self.training_data_shuffle_buffer_size,
                                                                  reshuffle_each_iteration=True   # controls whether the shuffle order should be different for each epoch
                                                                  )

        if self.training_data_batch_size is not None:
            # Combines consecutive elements of this dataset into batches
            self.training_dataset = self.training_dataset.batch(self.training_data_batch_size)

        # Most dataset input pipelines should end with a call to `prefetch`. This
        # allows later elements to be prepared while the current element is being
        # processed. This often improves latency and throughput, at the cost of
        # using additional memory to store prefetched elements.
        if self.enable_data_prefetch:
            self.training_dataset = self.training_dataset.prefetch(self.data_prefetch_size)

    def build_validation_dataset(self):
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(list(self.validation_filenames))
        self.validation_dataset = self.validation_dataset.map(
            (lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
            num_parallel_calls=self.num_parallel_calls)

        self.validation_dataset = self.validation_dataset.batch(self.training_data_batch_size)
        print(f"Validation on {len(self.validation_filenames)} images ")

    def read_image(self, path_img):
        output_session = boto3.Session()
        aws_session = AWSSession(output_session)
        rasterio_env = rasterio.Env(
            session=aws_session,
            GDAL_DISABLE_READDIR_ON_OPEN='NO',
            CPL_VSIL_CURL_USE_HEAD='NO',
            GDAL_GEOREF_SOURCES='INTERNAL',
            GDAL_TIFF_INTERNAL_MASK='NO'
        )
        with rasterio_env as env:
            path_to_s3_img = "s3://" + self.bucket_name + "/" + path_img.numpy().decode()
            with rasterio.open(path_to_s3_img, mode='r', sharing=False, GEOREF_SOURCES='INTERNAL') as src:
                if self.bands == ['all']:
                    # Need to transpose the image to have the channels last and not first as rasterio read the image
                    # The input for the model is width*height*channels
                    train_img = np.transpose(src.read(), (1, 2, 0))
                else:
                    train_img = np.transpose(src.read(self.bands), (1, 2, 0))
            # Normalize image
        train_img = tf.image.convert_image_dtype(train_img, tf.float32)
        return train_img

    def get_label_from_csv(self, path_img):
        # testing if path in the training csv file or in the val one
        if path_img.numpy().decode() in self.labels_file_train.paths.to_list():
            ### Training csv
            # path_img is a tf.string and needs to be converted into a string using .numpy().decode()
            id = int(self.labels_file_train[self.labels_file_train.paths == path_img.numpy().decode()].index.values)
            # The list of labels (e.g [0,1,0,0,0,0,0,0,0,0] is grabbed from the csv file on the row where the s3 path is
            label = self.labels_file_train.drop('paths', 1).iloc[int(id)].to_list()
        else:
            ### Validation csv
            # path_img is a tf.string and needs to be converted into a string using .numpy().decode()
            id = int(self.labels_file_val[self.labels_file_val.paths == path_img.numpy().decode()].index.values)
            # The list of labels (e.g [0,1,0,0,0,0,0,0,0,0] is grabbed from the csv file on the row where the s3 path is
            label = self.labels_file_val.drop('paths', 1).iloc[int(id)].to_list()
        return label

    # Function used in the map() and returns the image and label corresponding to the file_path input
    def process_path(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        return img, label

    def process_path_train_set_augment_flip_left_right(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        img = tf.image.flip_left_right(img)
        return img, label

    def process_path_train_set_augment_flip_up_down(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        img = tf.image.flip_up_down(img)
        return img, label

    def process_path_train_set_augment_rot90(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        img = tf.image.rot90(img)
        return img, label

    def generate_class_weight(self, file_path, df):

        # generate class_weights dict to be used for class_weight attribute in model

        if not self.s3_file_paths:
            data = json.load(open(self.label_mapping_path))
        else:
            data = json.load(self.label_mapping_path)
        num_labels = len(data['label_names'].keys())
        labels = {}
        no_label_class = []
        for column in df.columns[0:num_labels]:
            try:
                col_count = df[column].value_counts()[1]
                labels[column] = col_count
            except:
                no_label_class.append(column)

        print(f"The file {file_path} is missing positive labels for classes {no_label_class}")

        labels_sum = sum(labels.values())
        keys_len = len(labels.keys())

        class_weight = {}
        for label_name in labels.keys():
            class_weight[int(label_name)] = (1 / labels[label_name]) * (labels_sum) / keys_len

        return class_weight

    class SaveCheckpoints(keras.callbacks.Callback):

        epoch_save_list = None
        checkpoints_dir = None

        def __init__(self, base_name_checkpoint):
            self.base_name_checkpoint = base_name_checkpoint # s3 ?
            print("base_name_checkpoint:",self.base_name_checkpoint)

        def on_epoch_end(self, epoch, logs={}):
            print(f'\nEpoch {epoch} saving checkpoint')
            model_name = f'{self.base_name_checkpoint}_epoch_{epoch}.h5'
            self.model.save_weights(model_name, save_format='h5')
            s3 = boto3.resource('s3')
            BUCKET = "canopy-production-ml-output"
            s3.Bucket(BUCKET).upload_file(model_name, "ckpt/" + model_name)


if __name__ == '__main__':
    gen = DataLoader(label_file_path_train="labels_test_v1.csv",
                     label_file_path_val="val_labels.csv",
                     bucket_name='canopy-production-ml',
                     s3_file_paths=False,
                     data_extension_type='.tif',
                     training_data_shape=(100, 100, 18),
                     augment=False,
                     flip_left_right=True,
                     flip_up_down=True,
                     rot90=True,
                     enable_shuffle=True,
                     # training_data_shuffle_buffer_size=10,
                     training_data_batch_size=20,
                     # num_parallel_calls=int(2),
                     enable_data_prefetch=True)


    def Simple_CNN(numclasses, input_shape):
        model = Sequential([
            layers.Input(input_shape),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(numclasses)
        ])
        return model

    def define_model(numclasses, input_shape):
        # parameters for CNN
        input_tensor = Input(shape=input_shape)

        # introduce a additional layer to get from bands to 3 input channels
        input_tensor = Conv2D(3, (1, 1))(input_tensor)

        base_model_resnet50 = keras.applications.ResNet50(include_top=False,
                                                          weights='imagenet',
                                                          input_shape=(100, 100, 3))
        base_model = keras.applications.ResNet50(include_top=False,
                                                 weights=None,
                                                 input_tensor=input_tensor)

        for i, layer in enumerate(base_model_resnet50.layers):
            # we must skip input layer, which has no weights
            if i == 0:
                continue
            base_model.layers[i + 1].set_weights(layer.get_weights())

        # add a global spatial average pooling layer
        top_model = base_model.output
        top_model = GlobalAveragePooling2D()(top_model)

        # let's add a fully-connected layer
        top_model = Dense(2048, activation='relu')(top_model)
        top_model = Dense(2048, activation='relu')(top_model)
        # and a logistic layer
        predictions = Dense(numclasses, activation='sigmoid')(top_model)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        #     model = model.layers[-1].bias.assign([0.0]) # WIP getting an error ValueError: Cannot assign to variable dense_8/bias:0 due to variable shape (10,) and value shape (1,) are incompatible

        # model.summary()
        return model

    random_id = random.randint(1, 10001)
    wandb.init(project='project_canopy', sync_tensorboard=True)
    config = wandb.config
    base_name_checkpoint = "model_resnet"
    save_checkpoint_wandb = SaveCheckpoints(base_name_checkpoint)
    # model_checkpoint_callback_loss = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f'checkpoint_loss_{random_id}.h5',
    #     format='h5',
    #     verbose=1,
    #     save_weights_only=True,
    #     monitor='val_loss',
    #     mode='min',
    #     save_best_only=True)
    #
    # model_checkpoint_callback_recall = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f'checkpoint_recall_{random_id}.h5',
    #     format='h5',
    #     verbose=1,
    #     save_weights_only=True,
    #     monitor='val_recall',
    #     mode='max',
    #     save_best_only=True)
    #
    # model_checkpoint_callback_precision = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f'checkpoint_precision_{random_id}.h5',
    #     format='h5',
    #     verbose=1,
    #     save_weights_only=True,
    #     monitor='val_precision',
    #     mode='max',
    #     save_best_only=True)

    # reducelronplateau = tf.keras.callbacks.ReduceLROnPlateau(
    #   monitor='val_loss', factor=0.1, patience=5, verbose=1,
    #   mode='min', min_lr=0.000001)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_precision', mode='max', patience=20, verbose=1)

    callbacks_list = [save_checkpoint_wandb, early_stop]

    model = define_model(10, (100,100,18))

    # loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # Computes the cross-entropy loss between true labels and predicted labels.
    # Focal loss instead of class weights: https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    model.compile(loss=SigmoidFocalCrossEntropy(),
                  # https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
                  optimizer=keras.optimizers.Adam(0.001),
                  metrics=[tf.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           # Computes the precision of the predictions with respect to the labels.
                           tf.keras.metrics.Recall(name='recall'),
                           # Computes the recall of the predictions with respect to the labels.
                           F1Score(num_classes=10, name="f1_score")
                           # https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
                           ]
                  )

    # model = Simple_CNN(10, input_shape=(100, 100, 18))
    epochs = 2
    history = model.fit(gen.training_dataset, validation_data=gen.validation_dataset,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        # shuffle=True  # whether to shuffle the training data before each epoch
                        )
