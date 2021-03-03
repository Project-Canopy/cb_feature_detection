import os
import multiprocessing
import rasterio
import tensorflow as tf
from glob import glob
import pickle
import numpy as np
import os
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import keras
import pandas as pd
import boto3
import io

class DataLoader:
    def __init__(self, label_file_path_train="labels_test_v1.csv",
                 label_file_path_val="labels_val.csv",
                 bucket_name='canopy-production-ml',
                 data_extension_type='.tif',
                 bands=['all'],
                 training_data_shape=(100, 100, 18),
                 shuffle_and_repeat=False,
                 enable_just_shuffle=True,
                 enable_just_repeat=False,
                 training_data_shuffle_buffer_size=1000,
                 data_repeat_count=None,
                 training_data_batch_size=3,
                 normalization_value=255.0,
                 training_data_type=tf.float32,
                 label_data_type=tf.uint8,
                 enable_data_prefetch=True,
                 data_prefetch_size=2,
                 num_parallel_calls=2 * multiprocessing.cpu_count(),
                 output_shape=(tf.float32, tf.uint8)):
        #TODO add data augmentation e.g https://albumentations.ai/docs/examples/tensorflow-example/

        self.label_file_path_train = label_file_path_train
        self.labels_file_train = pd.read_csv(self.label_file_path_train)
        self.training_filenames = self.labels_file_train.paths.to_list()

        self.label_file_path_val = label_file_path_val
        self.labels_file_val = pd.read_csv(self.label_file_path_val)
        self.validation_filenames = self.labels_file_val.paths.to_list()
        self.bands = bands
        self.bucket_name = bucket_name

        self.enable_just_shuffle = enable_just_shuffle
        self.enable_just_repeat = enable_just_repeat
        self.data_repeat_count = data_repeat_count
        self.enable_data_prefetch = enable_data_prefetch
        self.shuffle_and_repeat = shuffle_and_repeat
        self.num_parallel_calls = num_parallel_calls
        self.data_prefetch_size = data_prefetch_size
        self.data_extension_type = data_extension_type
        self.output_shape = output_shape

        self.training_data_shape = training_data_shape
        self.training_data_shuffle_buffer_size = \
            training_data_shuffle_buffer_size \
            if training_data_shuffle_buffer_size is not None \
            else len(self.training_filenames) // training_data_batch_size

        self.training_data_batch_size = training_data_batch_size
        self.normalization_value = normalization_value
        self.training_data_type = training_data_type
        self.label_data_type = label_data_type

        self.build_training_dataset()
        self.build_validation_dataset()

    def build_training_dataset(self):
        self.training_dataset = tf.data.Dataset.from_tensor_slices(self.training_filenames)
        self.training_dataset = self.training_dataset.map((lambda x: tf.py_function(self.process_path, [x], self.output_shape)), num_parallel_calls=self.num_parallel_calls)

        if self.enable_just_shuffle is True:
            if self.shuffle_and_repeat is False:
                self.training_dataset = self.training_dataset.shuffle(self.training_data_shuffle_buffer_size)
            else:
                self.logger.warning(
                    ('Can not enable just shuffling of dataset because shuffle and repeat enabled'))

        if self.enable_just_repeat is True:
            if self.shuffle_and_repeat is False:
                self.training_dataset = self.training_dataset.repeat(
                    count=self.data_repeat_count)
            else:
                self.logger.warning(
                    ('Can not enable just repeat of dataset because shuffle and repeat enabled'))

        if self.shuffle_and_repeat is True:
            self.training_dataset = self.training_dataset.shuffle(self.training_data_shuffle_buffer_size)
            self.training_dataset = self.training_dataset.repeat(count=self.data_repeat_count)

        if self.training_data_batch_size is not None:
            self.training_dataset = self.training_dataset.batch(self.training_data_batch_size)

        if self.enable_data_prefetch is True:
            self.training_dataset = self.training_dataset.prefetch(self.data_prefetch_size)

    def build_validation_dataset(self):
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(self.validation_filenames)
        # self.validation_dataset = self.validation_dataset.map((lambda x: tf.py_function(self.process_path, [x], (tf.float32, tf.uint8))), num_parallel_calls=self.num_parallel_calls)
        self.validation_dataset = self.validation_dataset.map((lambda x: tf.py_function(self.process_path, [x], self.output_shape)), num_parallel_calls=self.num_parallel_calls)

        self.validation_dataset = self.validation_dataset.batch(self.training_data_batch_size)
        self.validation_dataset = self.validation_dataset.prefetch(self.data_prefetch_size)

    def read_image(self, path_img):
        s3 = boto3.resource('s3')
        obj = s3.Object(self.bucket_name, path_img.numpy().decode())
        obj_bytes = io.BytesIO(obj.get()['Body'].read())
        with rasterio.open(obj_bytes) as src:
            if self.bands == ['all']:
                train_img = np.transpose(src.read(), (1, 2, 0))
            else:
                train_img = np.transpose(src.read(bands), (1, 2, 0))
        train_img = tf.convert_to_tensor(train_img / self.normalization_value, dtype=self.training_data_type)
        return train_img

    def get_label_from_csv(self, path_img):
        if path_img.numpy().decode() in self.labels_file_train.paths.to_list():
            # Training csv
            id = int(self.labels_file_train[self.labels_file_train.paths == path_img.numpy().decode()].index.values)
        else:
            # Validation csv
            id = int(self.labels_file_val[self.labels_file_val.paths == path_img.numpy().decode()].index.values)

        label = self.labels_file_train.drop('paths', 1).iloc[int(id)].to_list()
        return label

    def process_path(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        return img, label

if __name__ == '__main__':
    gen = DataLoader(label_file_path_train="labels_test_v1.csv",
                        label_file_path_val="val_labels.csv",
                        bucket_name='canopy-production-ml',
                        data_extension_type='.tif',
                        training_data_shape=(100, 100, 18),
                        shuffle_and_repeat=False,
                        enable_just_shuffle=True,
                        enable_just_repeat=False,
                        training_data_shuffle_buffer_size=10,
                        data_repeat_count=None,
                        training_data_batch_size=20,
                        normalization_value=255.0,  #normalization
                        training_data_type=tf.float32,
                        label_data_type=tf.uint8,
                        enable_data_prefetch=False,
                        data_prefetch_size=tf.data.experimental.AUTOTUNE,
                        num_parallel_calls=int(2))

    no_of_val_imgs = len(gen.validation_filenames)
    no_of_train_imgs = len(gen.training_filenames)
    print("Validation on {} images ".format(str(no_of_val_imgs)))
    print("Training on {} images ".format(str(no_of_train_imgs)))

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

    def Resnet50(numclasses, input_shape):
        model = Sequential()
        model.add(keras.applications.ResNet50(include_top=False, pooling='avg', weights=None, input_shape=input_shape))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(2048, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(numclasses, activation='softmax'))
        model.layers[0].trainable = True
        return model

    model = Resnet50(10, input_shape=(100, 100, 18))
    # model = Simple_CNN(10, input_shape=(100, 100, 18))
    callbacks_list = []

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam())

    epochs = 10
    history = model.fit(gen.training_dataset, validation_data=gen.validation_dataset, epochs=epochs)
