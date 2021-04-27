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
from tensorflow_addons.metrics import F1Score, HammingLoss
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import random
import time

    
    
def download_model(model_url,weights_url):

    s3 = boto3.resource('s3')
    model_filename = "model.h5"
    model_weights_filename = "model_weights.h5"

    #Download Model, Weights

    bucket = model_url.split("/")[2]
    model_key = "/".join(model_url.split("/")[3:])
    s3.Bucket(bucket).download_file(model_key, model_filename)
    weights_key = "/".join(weights_url.split("/")[3:])
    s3.Bucket(bucket).download_file(weights_key, model_weights_filename)

    return 

    
def load_model(model_path,model_weights_path):

    model = tf.keras.models.load_model(model_path)
    model.load_weights(model_weights_path)

    return model 

def gen_timestamp():
    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    return time_stamp

def save_to_s3(output_dict,local_path,job_name,timestamp):
    
    with open(local_path, 'w') as jp:
        json.dump(output_dict, jp)
        
    filename = local_path.split("/")[-1]
    output_base_path = "s3://canopy-production-ml/inference/output/"
    job_name = "inference_output_test"
    full_name = f'{job_name}-{timestamp}.json'
    s3_path = "/".join(output_base_path.split("/")[3:]) + full_name
    BUCKET = output_base_path.split("/")[2]
    s3 = boto3.resource('s3')
    s3.Bucket(BUCKET).upload_file(local_path, s3_path)
    
def read_json_label_file():
    # to create
    
def output_sample_chips(json_file,label_name,amount_of_chips):
    # to do
    


def read_image_tf_out(window_arr):

    #create copy of bands list, remove ndvi band from copy 
    window_arr_no_ndvi = window_arr.copy()
    window_arr_no_ndvi = window_arr_no_ndvi[:-1] 
    tf_img_no_ndvi = tf.image.convert_image_dtype(window_arr_no_ndvi, tf.float32)
    
    ndvi_band = window_arr_no_ndvi[-1]
    tf_img_ndvi = tf.image.convert_image_dtype(ndvi_band, tf.float32)
    
    tf_img = tf.concat([tf_img_no_ndvi,[tf_img_ndvi]],axis=0)
    tf_img = tf.transpose(tf_img,perm=[1, 2, 0])
    tf_img = tf.expand_dims(tf_img, axis=0)

    return tf_img

def read_image(window_arr):

    #create copy of bands list, remove ndvi band from copy 
    window_arr_no_ndvi = window_arr.copy()
    window_arr_no_ndvi = window_arr_no_ndvi[:-1].astype('float32')
    
    ndvi_band = window_arr_no_ndvi[-1].astype('float32')
    
    img = np.concatenate([window_arr_no_ndvi,[ndvi_band]],axis=0)
    img = np.transpose(img, (1, 2, 0))


    return np.array([img])



if __name__ == '__main__':

    bands=[2, 3, 4, 8,11,12,18]
    input_shape_RGBNIRSWR1SWR2NDVI = (100,100,len(bands))
                                        
                                        

