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
import json


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

        self.class_weight_train = self.generate_class_weight(self.label_file_path_train)
        self.class_weight_val = self.generate_class_weight(self.label_file_path_val)

        self.build_training_dataset()
        self.build_validation_dataset()

    def read_s3_obj(self, s3_key):
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
            print("Training on {} images ".format(len(self.training_filenames * 2)))
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
        self.validation_dataset = self.validation_dataset.map(
            (lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
            num_parallel_calls=self.num_parallel_calls)

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

    # def get_weights(self, class_weight):
    #     weights = np.zeros(10)
    #     for id in range(len(class_weight)):
    #         weights[list(class_weight.keys())[id]] = list(class_weight.values())[id]
    #     return weights

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

    def generate_class_weight(self, file_path):

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

        print(f"The file {file_path} is missing positive labels for classes {no_label_class}")

        labels_sum = sum(labels.values())
        keys_len = len(labels.keys())

        class_weight = {}
        for label_name in labels.keys():
            class_weight[int(label_name)] = (1 / labels[label_name]) * (labels_sum) / keys_len

        return class_weight


if __name__ == '__main__':
    gen = DataLoader(label_file_path_train="labels_test_v1.csv",
                     label_file_path_val="val_labels.csv",
                     bucket_name='canopy-production-ml',
                     s3_file_paths=False,
                     data_extension_type='.tif',
                     training_data_shape=(100, 100, 18),
                     augment=True,
                     random_flip_up_down=True,
                     random_flip_left_right=True,
                     flip_left_right=True,
                     flip_up_down=True,
                     rot90=True,
                     transpose=False,
                     enable_shuffle=True,
                     training_data_shuffle_buffer_size=10,
                     training_data_batch_size=20,
                     training_data_type=tf.float32,
                     label_data_type=tf.uint8,
                     num_parallel_calls=int(2))


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
