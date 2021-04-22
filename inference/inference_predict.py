import os
import multiprocessing
import rasterio
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import tensorflow.keras as keras
import pandas as pd
import boto3
import io
import json
import random

    
    
def load_model(model_url,weights_url):

    s3 = boto3.resource('s3')
    model_filename = "model.h5"
    model_weights_filename = "model_weights.h5"

    #Download Model, Weights

    bucket = model_url.split("/")[2]
    model_key = "/".join(model_url.split("/")[3:])
    s3.Bucket(bucket).download_file(model_key, model_filename)
    weights_key = "/".join(weights_url.split("/")[3:])
    s3.Bucket(bucket).download_file(weights_key, model_weights_filename)


    model = tf.keras.models.load_model(model_filename)
    model.load_weights(model_weights_filename)

    return model 


def read_image(window_arr):

    if self.file_mode == "file":

        path_to_img = self.local_path_train + "/" + path_img.numpy().decode()

        if 18 in self.bands:

            #create copy of bands list, remove ndvi band from copy 
            bands_copy = self.bands.copy()
            bands_copy.remove(18)
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



if __name__ == '__main__':

    bands=[2, 3, 4, 8,11,12,18]
    input_shape_RGBNIRSWR1SWR2NDVI = (100,100,len(bands))
                                        
                                        

