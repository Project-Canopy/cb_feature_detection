import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers

import numpy as np
import argparse, os, subprocess, sys
import tensorflow.keras as keras
import pandas as pd
import io
import json
import random
import h5py
import multiprocessing
import time

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
subprocess.call([sys.executable, "nvidia-smi"])

# Script mode doesn't support requirements.txt
# Here's the workaround:
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


class DataLoader:
    def __init__(self, label_file_path_train="labels_test_v1.csv",
                 label_file_path_val="labels_val.csv",
                 bucket_name='margaux-bucket',
                 data_extension_type='.tif',
                 bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
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

        self.bucket_name = bucket_name

        self.label_file_path_train = label_file_path_train
        self.label_file_path_val = label_file_path_val

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

        self.training_data_batch_size = training_data_batch_size

        self.build_training_dataset()
        self.build_validation_dataset()

    def read_s3_obj(self, s3_key):
        s3 = boto3.resource('s3')
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
                                                                  reshuffle_each_iteration=True
                                                                  # controls whether the shuffle order should be different for each epoch
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


class SaveCheckpoints(keras.callbacks.Callback):
    epoch_save_list = None
    checkpoints_dir = None

    def __init__(self, base_name_checkpoint, random_id_training, bucket):
        self.base_name_checkpoint = base_name_checkpoint  # s3 ?
        self.bucket = bucket
        print("base_name_checkpoint:", self.base_name_checkpoint)

    def on_epoch_end(self, epoch, logs={}):
        print(f'\nEpoch {epoch} saving checkpoint')
        model_name = f'{self.base_name_checkpoint}_epoch_{epoch}.h5'
        self.model.save_weights(model_name, save_format='h5')
        s3 = boto3.resource('s3')
#         BUCKET = "margaux-bucket-us-east-1"
        checkpoint_s3_uri = f'ckpt/{random_id_training}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())}-{model_name}'
        s3.Bucket(self.bucket).upload_file(model_name, checkpoint_s3_uri)


if __name__ == '__main__':
    print("TensorFlow version", tf.__version__)
    print("Keras version", keras.__version__)
    # Keras-metrics brings additional metrics: precision, recall, f1
    install('keras-metrics')
    import keras_metrics

    install('tensorflow-addons')
    from tensorflow_addons.metrics import F1Score, HammingLoss
    from tensorflow_addons.losses import SigmoidFocalCrossEntropy

    install('wandb')
    import wandb
    from wandb.keras import WandbCallback

    install('rasterio')
    import rasterio
    from rasterio.session import AWSSession

    install('boto3')
    import boto3

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--augment', type=str, default="False")
    parser.add_argument('--flip_left_right', type=str, default="False")
    parser.add_argument('--flip_up_down', type=str, default="False")
    parser.add_argument('--rot90', type=str, default="False")
    parser.add_argument('--numclasses', type=float, default=10)
    parser.add_argument('--bands', required=True),
    parser.add_argument('--bucket', type=str, default="margaux-bucket-us-east-1"),
    parser.add_argument('--training_file', type=str, default="labels_test_v1.csv"),
    parser.add_argument('--validation_file', type=str, default="val_labels.csv"),

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
#     parser.add_argument('--gpu-count', type=int, default=0)
#     parser.add_argument('--model-dir', type=str, default="./")
#     parser.add_argument('--training', type=str, default="./")
#     parser.add_argument('--validation', type=str, default="./")

    args, _ = parser.parse_known_args()

    os.environ["WANDB_API_KEY"] = "6607ed7a49b452c2f3494ce60f9514f6c9e3b4e6"
    wandb.init(project='project_canopy')  # , sync_tensorboard=True
    config = wandb.config

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    bucket = args.bucket
    training_file = args.training_file
    validation_file = args.validation_file
    # Had to use strings because of the way sagemaker script was passing the args values 
    if args.augment.lower() == "false":
        augmentation_data = False
    else:
        args.augment = True
    if args.flip_left_right.lower() == "false":
        aug_flip_left_right = False
    else:
        aug_flip_left_right = True
    if args.flip_up_down.lower() == "false":
        aug_flip_up_down = False
    else:
        aug_flip_up_down = True
    if args.rot90 == "false":
        aug_rot90 = False
    else:
        aug_rot90 = True
    print(f"lr: {lr}, batch_size: {batch_size}, augmentation_data: {augmentation_data} " + 
          f"aug_flip_left_right {aug_flip_left_right}, aug_flip_up_down {aug_flip_up_down}, aug_rot90: {aug_rot90}")
    
    bands = args.bands.split(" ")
    bands = list(map(int, bands))   # convert to int
    print(f"bands {bands}")
    input_shape = (100, 100, int(len(bands)))
    print(f"Input shape: {input_shape}")

    numclasses = int(args.numclasses)

#     gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    validation_dir = args.validation
    print(f"training_dir {os.path.join(training_dir, training_file)}")
    print(f"val_dir {os.path.join(validation_dir, validation_file)}")

    print("Dataloader initialization...")
    gen = DataLoader(label_file_path_train=os.path.join(training_dir, training_file),
                     label_file_path_val=os.path.join(validation_dir, validation_file),
                     bucket_name=bucket,
                     data_extension_type='.tif',
                     bands=bands,
                     augment=augmentation_data,
                     flip_left_right=aug_flip_left_right,
                     flip_up_down=aug_flip_up_down,
                     rot90=aug_rot90,
                     enable_shuffle=True,
                     training_data_batch_size=batch_size,
                     enable_data_prefetch=True)


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


    base_name_checkpoint = "model_resnet"
    random_id_training = random.randint(1, 10001)
    save_checkpoint_wandb = SaveCheckpoints(base_name_checkpoint, random_id_training, bucket)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_precision', mode='max', patience=20, verbose=1)

    callbacks_list = [save_checkpoint_wandb, early_stop, WandbCallback()]

    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3","/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7"])

    with mirrored_strategy.scope():
        model = define_model(numclasses, input_shape)

        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # Computes the cross-entropy loss between true labels and predicted labels.
        # Focal loss instead of class weights: https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
        model.compile(loss=SigmoidFocalCrossEntropy(),
                      # https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
                      optimizer=keras.optimizers.Adam(lr),
                      metrics=[tf.metrics.BinaryAccuracy(name='accuracy'),
                               tf.keras.metrics.Precision(name='precision'),
                               # Computes the precision of the predictions with respect to the labels.
                               tf.keras.metrics.Recall(name='recall'),
                               # Computes the recall of the predictions with respect to the labels.
                               F1Score(num_classes=numclasses, name="f1_score")
                               # https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
                               ]
                      )

    history = model.fit(gen.training_dataset,
                        validation_data=gen.validation_dataset,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        )
