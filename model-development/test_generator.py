import os
import multiprocessing
import rasterio
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import tensorflow.keras as keras
# import keras
import pandas as pd
import boto3
import io
import json
# from tensorflow_addons.metrics import F1Score, HammingLoss
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import random
# from save_checkpoint_callback_custom import SaveCheckpoints
# import wandb
# from wandb.keras import WandbCallback
from rasterio.session import AWSSession

class TestGenerator:
    def __init__(self, 
                 training_dir = "./fsx",
label_file_path_test="labels_test_set.csv",
                 bucket_name='canopy-production-ml',
                 label_mapping_path="new_labels.json",
                 data_extension_type='.tif',
                 bands=['all'],
                 file_mode="s3",
                 test_data_shape=(100, 100, 18),
                 test_data_batch_size=20,
                 enable_data_prefetch=False,
                 data_prefetch_size=tf.data.experimental.AUTOTUNE,
                 num_parallel_calls=2 * multiprocessing.cpu_count(),
                 output_shape=(tf.float32, tf.float32)):
        
        

            

        self.label_file_path_test = label_file_path_test
        self.label_mapping_path = label_mapping_path
        self.labels_file_test = pd.read_csv(self.label_file_path_test)
        self.test_filenames = self.labels_file_test.paths.to_list()
        self.bands = bands
        self.file_mode = file_mode
        
        if self.file_mode == "s3":
            self.s3 = boto3.resource('s3')
            self.bucket_name = bucket_name
            
        if self.file_mode == "file":
            self.local_path_train = training_dir

        self.num_parallel_calls = num_parallel_calls
        self.enable_data_prefetch = enable_data_prefetch
        self.data_prefetch_size = data_prefetch_size
        self.data_extension_type = data_extension_type
        self.output_shape = output_shape

        self.test_data_shape = test_data_shape

        self.test_data_batch_size = test_data_batch_size

        self.class_weight_test = self.generate_class_weight(self.label_file_path_test, self.labels_file_test)

        self.build_testing_dataset()

#     def read_s3_obj(self, s3_key):
#         s3 = self.s3
#         obj = s3.Object(self.bucket_name, s3_key)
#         obj_bytes = io.BytesIO(obj.get()['Body'].read())
#         return obj_bytes

    def build_testing_dataset(self):
        # Tensor of all the paths to the images
        self.test_dataset = tf.data.Dataset.from_tensor_slices(self.test_filenames)

        self.test_dataset = self.test_dataset.map((
            lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
            num_parallel_calls=self.num_parallel_calls)
        self.length_testing_dataset = len(self.test_filenames)
        print(f"test on {self.length_testing_dataset} images")

        if self.test_data_batch_size is not None:
            # Combines consecutive elements of this dataset into batches
            self.test_dataset = self.test_dataset.batch(self.test_data_batch_size)

        # Most dataset input pipelines should end with a call to `prefetch`. This
        # allows later elements to be prepared while the current element is being
        # processed. This often improves latency and throughput, at the cost of
        # using additional memory to store prefetched elements.
        if self.enable_data_prefetch:
            self.test_dataset = self.test_dataset.prefetch(self.data_prefetch_size)

    def read_image(self, path_img):
        
        if self.file_mode == "s3":
            
            print("please use 'file' for the file mode. s3 not supported")
            
#             output_session = boto3.Session()
#             aws_session = AWSSession(output_session)
#             rasterio_env = rasterio.Env(
#                 session=aws_session,
#                 GDAL_DISABLE_READDIR_ON_OPEN='NO',
#                 CPL_VSIL_CURL_USE_HEAD='NO',
#                 GDAL_GEOREF_SOURCES='INTERNAL',
#                 GDAL_TIFF_INTERNAL_MASK='NO'
#             )
#             with rasterio_env as env:
#                 path_to_s3_img = "s3://" + self.bucket_name + "/" + path_img.numpy().decode()
#                 with rasterio.open(path_to_s3_img, mode='r', sharing=False, GEOREF_SOURCES='INTERNAL') as src:
#                     if self.bands == ['all']:
#                         # Need to transpose the image to have the channels last and not first as rasterio read the image
#                         # The input for the model is width*height*channels
#                         train_img = np.transpose(src.read(), (1, 2, 0))
#                     else:
#                         train_img = np.transpose(src.read(self.bands), (1, 2, 0))
                        
        if self.file_mode == "file":
            
            path_to_img = self.local_path_train + "/" + path_img.numpy().decode()
            
            if 18 in self.bands:

                #create copy of bands list, remove ndvi band from copy 
                bands_copy = self.bands.copy()
                bands_copy.remove(18)
                train_img_no_ndvi = rasterio.open(path_to_img).read(bands_copy)
                #normalize non_ndvi and ndvi bands separately, then combine as a single tensor (numpy) array
                train_img_no_ndvi = tf.image.convert_image_dtype(train_img_no_ndvi, tf.float32)
                ndvi_band = rasterio.open(path_to_img).read(18)
                train_img_ndvi = tf.image.convert_image_dtype(ndvi_band, tf.float32)
                train_img = tf.concat([train_img_no_ndvi,[train_img_ndvi]],axis=0)
                train_img = tf.transpose(train_img,perm=[1, 2, 0])

            else:

                train_img = np.transpose(rasterio.open(path_to_img).read(self.bands), (1, 2, 0))
                # Normalize image
                train_img = tf.image.convert_image_dtype(train_img, tf.float32)

            return train_img

    def get_label_from_csv(self, path_img):
        # testing if path in the test csv file or in the val one
        if path_img.numpy().decode() in self.labels_file_test.paths.to_list():
            ### test csv
            # path_img is a tf.string and needs to be converted into a string using .numpy().decode()
            id = int(self.labels_file_test[self.labels_file_test.paths == path_img.numpy().decode()].index.values[0])
            # The list of labels (e.g [0,1,0,0,0,0,0,0,0,0] is grabbed from the csv file on the row where the s3 path is
            label = self.labels_file_test.drop('paths', 1).iloc[int(id)].to_list()
        else:
            ### Validation csv
            # path_img is a tf.string and needs to be converted into a string using .numpy().decode()
            id = int(self.labels_file_val[self.labels_file_val.paths == path_img.numpy().decode()].index.values[0])
            # The list of labels (e.g [0,1,0,0,0,0,0,0,0,0] is grabbed from the csv file on the row where the s3 path is
            label = self.labels_file_val.drop('paths', 1).iloc[int(id)].to_list()
        return label

    # Function used in the map() and returns the image and label corresponding to the file_path input
    def process_path(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        return img, label

    def process_path_test_set_augment_flip_left_right(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        img = tf.image.flip_left_right(img)
        return img, label

    def process_path_test_set_augment_flip_up_down(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        img = tf.image.flip_up_down(img)
        return img, label

    def process_path_test_set_augment_rot90(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        img = tf.image.rot90(img)
        return img, label

    def generate_class_weight(self, file_path, df):

        # generate class_weights dict to be used for class_weight attribute in model
        data = json.load(open(self.label_mapping_path))
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
    gen = TestGenerator(label_file_path_test="labels_test_v1.csv",
                         bucket_name='canopy-production-ml',
                         label_mapping_path="labels.json",
                         data_extension_type='.tif',
                         bands=['all'],
                         test_data_shape=(100, 100, 18),
                         test_data_batch_size=20,
                         enable_data_prefetch=False,
                         data_prefetch_size=tf.data.experimental.AUTOTUNE,
                         num_parallel_calls=2 * multiprocessing.cpu_count(),
                         output_shape=(tf.float32, tf.float32))
