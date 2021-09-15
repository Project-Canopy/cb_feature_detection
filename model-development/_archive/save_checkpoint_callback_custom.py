import tensorflow.keras as keras
import os
import wandb
import io
import h5py
import boto3

class SaveCheckpoints(keras.callbacks.Callback):
    epoch_save_list = None
    checkpoints_dir = None

    def __init__(self, base_name_checkpoint):
        self.base_name_checkpoint = base_name_checkpoint # s3 ?

    def on_epoch_end(self, epoch, logs={}):
        print(f'\nEpoch {epoch} saving checkpoint')
        model_name = f'{self.base_name_checkpoint}-epoch{epoch}.h5'
        self.model.save_weights(model_name, save_format='h5')
        s3 = boto3.resource('s3')
        BUCKET = "canopy-production-ml-output"
        s3.Bucket(BUCKET).upload_file(model_name, "ckpt/" + model_name)

