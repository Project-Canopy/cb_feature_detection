import os
import rasterio as rio
import tensorflow as tf
import numpy as np
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras import layers
# import tensorflow.keras as keras
from rasterio.windows import Window
from rasterio.windows import get_data_window
from shapely.geometry import Polygon
from shapely.geometry import box
import pandas as pd
import boto3
import io
import json
from tensorflow_addons.metrics import F1Score, HammingLoss
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import random
import time
from glob import glob
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


class Handler:
    def __init__(self, config):
        """(Required) Called once during each worker initialization. Performs
        setup such as downloading/initializing the model or downloading a
        vocabulary.

        Args:
            tensorflow_client (required): TensorFlow client which is used to
                make predictions. This should be saved for use in handle_batch().
            config (required): Dictionary passed from API configuration (if
                specified) merged with configuration passed in with Job
                Submission API. If there are conflicting keys, values in
                configuration specified in Job submission takes precedence.
            job_spec (optional): Dictionary containing the following fields:
                "job_id": A unique ID for this job
                "api_name": The name of this batch API
                "config": The config that was provided in the job submission
                "workers": The number of workers for this job
                "total_batch_count": The total number of batches in this job
                "start_time": The time that this job started
        """
        self.job_name = config["job_name"]
        self.model_url = config["model_url"] 
        self.weights_url = config["weights_url"]
        self.model_filename = "model.h5"
        self.model_weights_filename = "model_weights.h5"
        self.download_model_files()
        self.model = self.load_model()
        self.output_dir = "./output/"
        self.label_list = ["Industrial_agriculture","ISL","Mining","Roads","Shifting_cultivation"]
        self.job_type = config["type"].lower()
        self.src_crs = "EPSG:" + str(config["source_crs"])
        if self.job_type == "realtime":
            self.s3_dir_url = config["file_dir"]
        if self.job_type == "batch":
            self.timestamp = config["timestamp"]
        self.full_name = f'{self.job_name}-{self.timestamp}.json'
        self.output_s3_base_path = "s3://canopy-production-ml/inference/output/"
        self.full_s3_path = self.output_s3_base_path + self.full_name
        self.key = "/".join(self.output_s3_base_path.split("/")[3:]) + self.full_name
        self.BUCKET = self.output_s3_base_path.split("/")[2]
        if not os.path.exists(self.output_dir):
            os.system(f"mkdir {self.output_dir}")
        self.local_file_path = self.output_dir + self.full_name
        
        
        print("model loaded and ready for predictions")

        # self.handle_batch()
        

    def handle_batch(self, payload):

        """(Required) Called once per batch. Preprocesses the batch payload (if
        necessary), runs inference (e.g. by calling
        self.client.predict(model_input)), postprocesses the inference output
        (if necessary), and writes the predictions to storage (i.e. S3 or a
        database, if desired).

        Args:
            payload (required): a batch (i.e. a list of one or more samples).
            batch_id (optional): uuid assigned to this batch.
        Returns:
            Nothing
        """
        print("received payload")

        if self.job_type == "realtime":
            if payload["start"].lower() == "none":
                print("starting real-time predictions")
                self.predict()
            else:
                print("to initiate real-time job pass in 'none' for the 'start' argument in your request")
        if self.job_type == "batch":
            print(payload)
            self.s3_paths = payload
            previous_dict = self.download_json()
            if previous_dict:
                current_id = payload[0].split("/")[-1].split("_")[0]
                if current_id in list(previous_dict.keys()):
                    print(f"granule id {current_id} is already in {self.full_name} on s3. skipping prediction.")
                    pass
                else:
                    print("continuing batch predictions")
                    self.predict()
            else:
                print("starting batch predictions")
                self.predict()


    def on_job_complete(self):
        """(Optional) Called once after all batches in the job have been
        processed. Performs post job completion tasks such as aggregating
        results, executing web hooks, or triggering other jobs.
        """
        pass



    
    def download_model_files(self):

        s3 = boto3.resource('s3')

        #Download Model, Weights

        bucket = self.model_url.split("/")[2]
        model_key = "/".join(self.model_url.split("/")[3:])
        s3.Bucket(bucket).download_file(model_key, self.model_filename)
        weights_key = "/".join(self.weights_url.split("/")[3:])
        s3.Bucket(bucket).download_file(weights_key, self.model_weights_filename)

        return 

        
    def load_model(self):

        model = tf.keras.models.load_model(self.model_filename)
        model.load_weights(self.model_weights_filename)

        return model 

    def s3_dir_ls(self,s3_dir_url):

        objs = []
        bucket = s3_dir_url.split("/")[2]
        key = "/".join(s3_dir_url.split("/")[3:5])

        s3 = boto3.resource('s3')
        my_bucket = s3.Bucket(bucket)


        for obj in my_bucket.objects.filter(Prefix=key):
            objs.append("s3://" + bucket + "/" + obj.key)

        return objs[1:]

    def download_json(self):

        try:
            s3 = boto3.resource('s3')
            s3.Bucket(self.BUCKET).download_file(self.key, self.local_file_path)
            with open(self.local_file_path) as j:
                json_data = json.load(j)
            return json_data
        except:
            print("no json file found")
            return None

    def gen_timestamp(self):
        time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        return time_stamp

    def save_to_s3(self,output_dict):

        s3 = boto3.resource('s3')

        if self.job_type == "batch" and self.full_s3_path in self.s3_dir_ls(self.output_s3_base_path):
            print(f"existing json record {self.full_name} for job found on s3. appending to existing file")
            json_data = self.download_json()   
            output_dict.update(json_data)

        with open(self.local_file_path, "w") as f:
            json.dump(output_dict, f)   
        
        s3.Bucket(self.BUCKET).upload_file(self.local_file_path, self.key)

    def read_image_tf_out(self,window_arr):

        #create copy of bands list, remove ndvi band from copy 
        window_arr_no_ndvi = window_arr.copy()
        window_arr_no_ndvi = window_arr_no_ndvi[:-1] 
        tf_img_no_ndvi = tf.image.convert_image_dtype(window_arr_no_ndvi, tf.float32)
        
        ndvi_band = window_arr[-1]
        tf_img_ndvi = tf.image.convert_image_dtype(ndvi_band, tf.float32)
        
        tf_img = tf.concat([tf_img_no_ndvi,[tf_img_ndvi]],axis=0)
        tf_img = tf.transpose(tf_img,perm=[1, 2, 0])
        tf_img = tf.expand_dims(tf_img, axis=0)

        return tf_img

    def get_windows(self,img_dim, patch_size=(100, 100), stride=(100, 100)):
        patch_size = np.array(patch_size)
        stride = np.array(stride)
        img_dim = np.array(img_dim)
        # to take into account edges, add additional blocks around right side edge and bottom edge of raster
        new_img_dim = [img_dim[0] + stride[0],img_dim[1] + stride[0]]
        
        max_dim = (new_img_dim//patch_size)*patch_size - patch_size

        ys = np.arange(0, img_dim[0], stride[0])
        xs = np.arange(0, img_dim[1], stride[1])

        tlc = np.array(np.meshgrid(ys, xs)).T.reshape(-1, 2)
        tlc = tlc[tlc[:, 0] <= max_dim[0]]
        tlc = tlc[tlc[:, 1] <= max_dim[1]]
        
        windows = []
        for y,x in tlc.astype(int):
            windows.append(Window(x, y, patch_size[1], patch_size[0]))

        return windows

    def add_ndvi(self,data, dtype_1=rio.float32):
        
        nir = data[3].astype(dtype_1)
        red = data[2].astype(dtype_1)
        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        # Calculate NDVI
        ndvi = ((nir - red) / (nir + red)).astype(dtype_1)
        # Rescaling for use in 16bit output
        ndvi = (ndvi + 1) * (2**15 - 1)
        # Add NDVI band to end of array    
        rast = np.concatenate((data,[ndvi]),axis=0)
        rast = rast.astype(rio.uint16)
        return rast

    def predict(self,patch_size=100,
                stride=100,SAVE=False,SAVE_INDIVIDUAL=False,
                bands=[2, 3, 4, 8, 11, 12], 
                predict_thresh=.5, 
                job_name=None, 
                output_filename=None):

        print("starting prediction function")
        model=self.model
        label_list=self.label_list
        # granule_list=glob(f'{granule_dir}/*.tif')
        if self.job_type == "realtime":
            granule_list = self.s3_dir_ls(self.s3_dir_url)
            timestamp = self.gen_timestamp()
        if self.job_type == "batch":
            print(self.s3_paths)
            granule_list = self.s3_paths
            timestamp = self.timestamp
        print(f"found {len(granule_list)} granules")
        output_dict={}
        for j,granule_path in enumerate(granule_list):
            granule_id = granule_path.split("/")[-1].split("_")[0]

            with rio.open(granule_path) as src:

                with WarpedVRT(src, crs=self.src_crs, resampling=Resampling.nearest) as vrt:

                    windows = self.get_windows(vrt.shape, (patch_size, patch_size), (stride, stride))

                    for i, window in enumerate(windows):
                        print(f"predicting window {i + 1} of {len(windows)} of granulate {j + 1} of {len(granule_list)}",end='\r', flush=True)
                        label_name_list = []
                        window_id = i+1
                        data = vrt.read(bands,window=window, masked=True)
                        data = self.add_ndvi(data)
                        shape = data.shape
                        new_shape = (data.shape[0],patch_size,patch_size)

                        if shape != new_shape:
                            filled_array = np.full(new_shape, 0)
                            filled_array[:shape[0],:shape[1],:shape[2]] = data
                            data = filled_array
                            window = Window(window.col_off,window.row_off,shape[2],shape[1])


                        #image pre-processing / inference
                        prediction = model.predict(self.read_image_tf_out(data))
                        prediction = np.where(prediction > predict_thresh, 1, 0)
                        prediction_i = np.where(prediction == 1)[1]
                        for i in prediction_i:
                            label_name_list.append(label_list[i])

                        #vectorizing raster bounds for visualization 
                        window_bounds = rio.windows.bounds(window, vrt.transform, height=patch_size, width=patch_size)
                        geom = box(*window_bounds)
                        geom_coords = list(geom.exterior.coords)
        #                 window_geom_list.append(geom)

                        #create or append to dict....

                        if granule_id in output_dict:
                            output_dict[granule_id].append({"window_id":window_id,"polygon_coords":geom_coords,"labels":label_name_list})
                        else:
                            output_dict[granule_id] = [{"window_id":window_id,"polygon_coords":geom_coords,"labels":label_name_list}]

                    self.save_to_s3(output_dict)


        return 