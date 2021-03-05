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

        self.label_file_path_train = label_file_path_train
        self.labels_file_train = pd.read_csv(self.label_file_path_train)
        self.training_filenames = self.labels_file_train.paths.to_list()

        self.label_file_path_val = label_file_path_val
        self.labels_file_val = pd.read_csv(self.label_file_path_val)
        self.validation_filenames = self.labels_file_val.paths.to_list()
        self.bands = bands
        self.bucket_name = bucket_name

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
            if self.shuffle_and_repeat is False:
                self.training_dataset = self.training_dataset.shuffle(self.training_data_shuffle_buffer_size)
            else:
                self.logger.warning(
                    ('Can not enable just shuffling of dataset because shuffle and repeat enabled'))

        if self.training_data_batch_size is not None:
            self.training_dataset = self.training_dataset.batch(self.training_data_batch_size)

        if self.enable_data_prefetch:
            self.training_dataset = self.training_dataset.prefetch(self.data_prefetch_size)

    def build_validation_dataset(self):
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(self.validation_filenames)
        self.validation_dataset = self.validation_dataset.map((lambda x: tf.py_function(self.process_path, [x], self.output_shape)), num_parallel_calls=self.num_parallel_calls)

        self.validation_dataset = self.validation_dataset.batch(self.training_data_batch_size)
        print("Validation on {} images ".format(len(self.validation_filenames)))

        # self.validation_dataset = self.validation_dataset.prefetch(self.data_prefetch_size)

    def read_image(self, path_img):
        s3 = boto3.resource('s3')
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

if __name__ == '__main__':
    gen = DataLoader(label_file_path_train="labels_test_v1.csv",
                        label_file_path_val="val_labels.csv",
                        bucket_name='canopy-production-ml',
                        data_extension_type='.tif',
                        training_data_shape=(100, 100, 18),
                        augment=True,
                        random_flip_up_down=True,
                        random_flip_left_right=True,
                        flip_left_right=True,
                        flip_up_down=True,
                        rot90=True,
                        transpose=False,
                        enable_shuffle=False,
                        training_data_shuffle_buffer_size=10,
                        training_data_batch_size=20,
                        training_data_type=tf.float32,
                        label_data_type=tf.uint8,
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
