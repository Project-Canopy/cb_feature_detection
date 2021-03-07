import os
import multiprocessing
import rasterio
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import keras
import pandas as pd
import boto3
import io
import json
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    #parser.add_argument('--epochs', type=int, default=1)
    #parser.add_argument('--batch_size', type=int, default=64)
    
    #parser.add_argument('--num_words', type=int)
    #parser.add_argument('--word_index_len', type=int)
    #parser.add_argument('--labels_index_len', type=int)
    #parser.add_argument('--embedding_dim', type=int)
    #parser.add_argument('--max_sequence_len', type=int)
    
    # data directories
    # parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    #parser.add_argument('--output', type=str, default=os.environ.get('SM_CHANNEL_OUTPUT'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--labels', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    
    # embedding directory
    #parser.add_argument('--embedding', type=str, default=os.environ.get('SM_CHANNEL_EMBEDDING'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    #parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()


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
                 training_data_batch_size=3,
                 training_data_type=tf.float32,
                 label_data_type=tf.uint8,
                 enable_data_prefetch=False,
                 data_prefetch_size=tf.data.experimental.AUTOTUNE,
                 num_parallel_calls=2 * multiprocessing.cpu_count(),
                 output_shape=(tf.float32, tf.uint8)):

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
        self.num_parallel_calls = num_parallel_calls
        self.enable_data_prefetch = enable_data_prefetch
        self.data_prefetch_size = data_prefetch_size
        self.data_extension_type = data_extension_type
        self.output_shape = output_shape

        self.training_data_shape = training_data_shape
        self.training_data_shuffle_buffer_size = \
            training_data_shuffle_buffer_size \
            if training_data_shuffle_buffer_size is not None \
            else len(self.training_filenames) // training_data_batch_size

        self.training_data_batch_size = training_data_batch_size
        self.training_data_type = training_data_type
        self.label_data_type = label_data_type

        self.build_training_dataset()
        self.build_validation_dataset()

        self.class_weight = self.generate_class_weight()


    def read_s3_obj(self,s3_key):
        s3 = self.s3
        obj = s3.Object(self.bucket_name, s3_key)
        obj_bytes = io.BytesIO(obj.get()['Body'].read())
        return obj_bytes


    def build_training_dataset(self):
        self.training_dataset = tf.data.Dataset.from_tensor_slices(self.training_filenames)

        if self.augment is True:
            print("Data augmentation enabled ")
            self.training_dataset_non_aug = self.training_dataset.map((
                lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
                num_parallel_calls=self.num_parallel_calls)
            self.training_dataset_aug = self.training_dataset.map((
                lambda x: tf.py_function(self.process_path_train_set_augment, [x], self.output_shape)),
                num_parallel_calls=self.num_parallel_calls)

            datasets = [self.training_dataset_non_aug,
                        self.training_dataset_aug]

            # Define a dataset containing `[0, 1, 0, 1, 0, 1, ..., 0, 1]`.
            choice_dataset = tf.data.Dataset.range(2).repeat(len(self.training_filenames))
            self.training_dataset = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
            print("Training on {} images ".format(len(self.training_filenames*2)))
        else:
            print("No data augmentation. Please set augment to True if you want to augment training dataset")
            self.training_dataset = self.training_dataset.map((
                    lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
                    num_parallel_calls=self.num_parallel_calls)
            print("Training on {} images ".format(len(self.training_filenames)))

        if self.enable_shuffle is True:
            self.training_dataset = self.training_dataset.shuffle(self.training_data_shuffle_buffer_size)

        if self.training_data_batch_size is not None:
            self.training_dataset = self.training_dataset.batch(self.training_data_batch_size)

        if self.enable_data_prefetch:
            self.training_dataset = self.training_dataset.prefetch(self.data_prefetch_size)

    def build_validation_dataset(self):
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(list(self.validation_filenames))
        self.validation_dataset = self.validation_dataset.map((lambda x: tf.py_function(self.process_path, [x], self.output_shape)), num_parallel_calls=self.num_parallel_calls)

        self.validation_dataset = self.validation_dataset.batch(self.training_data_batch_size)
        print("Validation on {} images ".format(len(self.validation_filenames)))

        # self.validation_dataset = self.validation_dataset.prefetch(self.data_prefetch_size)

    def read_image(self, path_img):
        s3 = self.s3
        obj = s3.Object(self.bucket_name, path_img.numpy().decode())
        obj_bytes = io.BytesIO(obj.get()['Body'].read())
        with rasterio.open(obj_bytes) as src:
            if self.bands == ['all']:
                train_img = np.transpose(src.read(), (1, 2, 0))
            else:
                train_img = np.transpose(src.read(bands), (1, 2, 0))
        # Normalize image
        train_img = tf.image.convert_image_dtype(train_img, tf.float32)
        return train_img

    def get_label_from_csv(self, path_img):
        if path_img.numpy().decode() in self.labels_file_train.paths.to_list():
            # Training csv
            id = int(self.labels_file_train[self.labels_file_train.paths == path_img.numpy().decode()].index.values)
            label = self.labels_file_train.drop('paths', 1).iloc[int(id)].to_list()
        else:
            # Validation csv
            id = int(self.labels_file_val[self.labels_file_val.paths == path_img.numpy().decode()].index.values)
            label = self.labels_file_val.drop('paths', 1).iloc[int(id)].to_list()
        return label

    def process_path(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        return img, label


    def process_path_train_set_augment(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        # TODO no all augmentations at once
        # apply simple augmentations
        if self.random_flip_up_down is True:
            img = tf.image.random_flip_up_down(img)
        if self.random_flip_left_right is True:
            img = tf.image.random_flip_left_right(img)
        if self.flip_left_right is True:
            img = tf.image.flip_left_right(img)
        if self.flip_up_down is True:
            img = tf.image.flip_up_down(img)
        if self.rot90 is True:
            img = tf.image.rot90(img)
        if self.transpose is True:
            img = tf.image.transpose(img)

        return img, label
    
    def generate_class_weight(self):

        # generate class_weights dict to be used for class_weight attribute in model
    
        df = self.labels_file_train
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
        
        print(f"Your training file is missing positive labels for classes {no_label_class}")

        labels_sum = sum(labels.values())
        keys_len = len(labels.keys())

        class_weight = {}
        for label_name in labels.keys():
            class_weight[int(label_name)] = (1 / labels[label_name]) * (labels_sum)/keys_len

        return class_weight

if __name__ == '__main__':

    args, _ = parse_args()
    batch_size = 20
    label_file_path_train = args.train
    label_file_path_val = args.val
    label_mapping_path = args.labels


    gen = DataLoader(label_file_path_train=label_file_path_train,
                    label_file_path_val=label_file_path_val,
                    label_mapping_path=label_mapping_path,
                    s3_file_paths=True,
                    bucket_name='canopy-production-ml',
                    data_extension_type='.tif',
                    training_data_shape=(100, 100, 18),
                    augment=True,
                    random_flip_up_down=False, #Randomly flips an image vertically (upside down). With a 1 in 2 chance, outputs the contents of `image` flipped along the first dimension, which is `height`.
                    random_flip_left_right=False,
                    flip_left_right=False,
                    flip_up_down=False,
                    rot90=True,
                    transpose=False,
                    enable_shuffle=False,
                    training_data_shuffle_buffer_size=10,
                    training_data_batch_size=batch_size,
                    training_data_type=tf.float32,
                    label_data_type=tf.uint8,
                    num_parallel_calls=int(2))

    def define_model(numclasses,input_shape):
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
            base_model.layers[i+1].set_weights(layer.get_weights())

        # add a global spatial average pooling layer
        top_model = base_model.output
        top_model = GlobalAveragePooling2D()(top_model)

        # let's add a fully-connected layer
        top_model = Dense(2048, activation='relu')(top_model)
        top_model = Dense(2048, activation='relu')(top_model)
        # and a logistic layer
        predictions = Dense(numclasses, activation='softmax')(top_model)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        model.summary()
        return model

    random_id = 1234 
    checkpoint_file = f"checkpoint_{random_id}.h5"

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= checkpoint_file,
    format='h5',
    verbose=1,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

    reducelronplateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=10, verbose=1,
    mode='min', min_lr=1e-10)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min', patience=20, verbose=1)

    callbacks_list = [model_checkpoint_callback, reducelronplateau, early_stop]

    model = define_model(10, (100,100,18))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          optimizer=keras.optimizers.Adam(),
                          metrics=[tf.metrics.BinaryAccuracy(name='accuracy')])

    epochs = 20
    history = model.fit(gen.training_dataset, validation_data=gen.validation_dataset, 
                        epochs=epochs, 
                        callbacks=callbacks_list)
