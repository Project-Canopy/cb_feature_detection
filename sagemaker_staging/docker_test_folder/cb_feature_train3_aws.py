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
import argparse
import wandb
from wandb.keras import WandbCallback
import random
from tensorflow_addons.metrics import F1Score, HammingLoss
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from data_loader import DataLoader


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--wandb_key', type=str)

    parser.add_argument('--bands', help='delimited list input', type=str, default='1,2,3,4,5,6,7,8,9,10,11,12,16,17,18')
    #parser.add_argument('--num_words', type=int)
    #parser.add_argument('--word_index_len', type=int)
    #parser.add_argument('--labels_index_len', type=int)
    #parser.add_argument('--embedding_dim', type=int)
    #parser.add_argument('--max_sequence_len', type=int)
    
    # data directories
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    parser.add_argument('--s3_chkpt_dir', type=str)
    parser.add_argument('--output', type=str, default="/opt/ml/output")
    parser.add_argument('--job_name', 
                        type=str, default=json.loads(os.environ.get('SM_TRAINING_ENV'))["job_name"])
#     parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
#     parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
#     parser.add_argument('--labels', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    
    # embedding directory
    #parser.add_argument('--embedding', type=str, default=os.environ.get('SM_CHANNEL_EMBEDDING'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    parser.add_argument('--starting_checkpoint', type=str, default=None)
    
    parser.add_argument('--augment', type=str2bool, default=True)
    parser.add_argument('--flip_left_right', type=str2bool, default=True)
    parser.add_argument('--flip_up_down', type=str2bool, default=True)
    parser.add_argument('--rot90', type=str2bool, default=True)
    
    return parser.parse_known_args()

    
def check_gpu():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    
class SaveCheckpoints(keras.callbacks.Callback):
    epoch_save_list = None
    checkpoints_dir = None

    def __init__(self, 
                base_name_checkpoint=None,
                lcl_chkpt_dir=None,
                s3_chkpt_dir=None):

        self.base_name_checkpoint = base_name_checkpoint # s3 ?
        self.lcl_chkpt_dir = lcl_chkpt_dir 
        self.s3_chkpt_dir = s3_chkpt_dir

    def on_epoch_end(self, epoch, logs={}):
        epoch = epoch + 1 
        print(f'\nEpoch {epoch} saving checkpoint')
        model_name = f'{self.base_name_checkpoint}_epoch_{epoch}.h5'
        local_path =  self.lcl_chkpt_dir + "/" + model_name
        s3_path = self.s3_chkpt_dir + "/" + model_name
        self.model.save_weights(local_path, save_format='h5')
        s3 = boto3.resource('s3')
        BUCKET = "canopy-production-ml-output"
        s3.Bucket(BUCKET).upload_file(local_path, s3_path)
        last_chkpt_filename = "last_chkpt.h5"
        last_chkpt_path = self.lcl_chkpt_dir + "/" + last_chkpt_filename
        self.model.save_weights(last_chkpt_path, save_format='h5')
        s3_path = self.s3_chkpt_dir + "/" + last_chkpt_filename
        s3.Bucket(BUCKET).upload_file(last_chkpt_path, s3_path)
        

if __name__ == '__main__':
    
    print(os.command("ls /opt/ml/data/fsx/"))


#     local_test = False
    
#     check_gpu()
    
#     args, _ = parse_args()
    
#     wandb_key = args.wandb_key
#     os.environ["WANDB_API_KEY"]=wandb_key
#     wandb.init(project='project_canopy', tensorboard=True)
#     config = wandb.config
    
#     job_name = args.job_name
#     data_dir = args.data
#     lcl_chkpt_dir = args.model_dir + "/" + "checkpoints"
#     s3_chkpt_base_dir = args.s3_chkpt_dir
#     epochs = args.epochs
#     batch_size=args.batch_size
#     bands = [int(band) for band in args.bands.split(',')]
#     starting_checkpoint = args.starting_checkpoint
#     print('Starting checkpoint:', starting_checkpoint)
    
#     if not os.path.exists(lcl_chkpt_dir):
#         os.mkdir(lcl_chkpt_dir)

#     files = ["labels_train.csv","labels_val.csv","labels.json"]


#     if local_test:

#         base_path = "/Users/purgatorid/Documents/GitHub/Project Canopy/cb_feature_detection/sagemaker_staging/"

#         label_file_path_train = base_path + files[0]
#         label_file_path_val = base_path + files[1]
#         label_mapping_path = base_path + files[2]
#         lcl_chkpt_dir = "./checkpoints"

#     else:

#         label_file_path_train = os.path.join(data_dir, files[0])
#         label_file_path_val = os.path.join(data_dir,files[1])
#         label_mapping_path = os.path.join(data_dir,files[2])

    
#     # Variables definition
#     config.batch_size = batch_size
#     config.learning_rate = 0.001
#     config.label_file_path_train=label_file_path_train # labels_1_4_train_v2
#     config.label_file_path_val=label_file_path_val
#     config.loss = SigmoidFocalCrossEntropy() # tf.keras.losses.BinaryCrossentropy(from_logits=False)
#     config.optimizer = keras.optimizers.Adam(config.learning_rate)
#     config.input_shape = (100, 100, len(bands))
#     config.numclasses=10
    
#     config.random_flip_up_down=False
#     config.random_flip_left_right=False
#     config.transpose=False
#     config.enable_shuffle=True
    
# #     config.augment = augment
# #     config.flip_left_right = flip_left_right
# #     config.flip_up_down = flip_up_down
# #     config.rot90 = rot90

#     print(f"batch size {config.batch_size}, learning_rate {config.learning_rate}, augment {config.augment}")
    

#     gen = DataLoader(label_file_path_train=config.label_file_path_train, # test labels_test_v1 TODO use s3 paths
#                     label_file_path_val=config.label_file_path_val, # or val_all
#                     label_mapping_path=label_mapping_path, 
#                     bucket_name='canopy-production-ml',
#                     data_extension_type='.tif',
#                     #training_data_shape=config.input_shape,
#                     augment=config.augment,
#                     s3_file_paths=False,
#                     flip_left_right=config.flip_left_right,
#                     flip_up_down=config.flip_up_down,
#                     rot90=config.rot90,
#                     transpose=config.transpose,
#                     enable_shuffle=config.enable_shuffle,
#                     training_data_batch_size=config.batch_size,
#                     enable_data_prefetch=True,
#                     bands=bands
#                     )
    
#     def define_model(numclasses,input_shape,starting_checkpoint=None,lcl_chkpt_dir=None):
#         # parameters for CNN
#         input_tensor = Input(shape=input_shape)

#         # introduce a additional layer to get from bands to 3 input channels
#         input_tensor = Conv2D(3, (1, 1))(input_tensor)

#         base_model_resnet50 = keras.applications.ResNet50(include_top=False,
#                                   weights='imagenet',
#                                   input_shape=(100, 100, 3))
#         base_model = keras.applications.ResNet50(include_top=False,
#                          weights=None,
#                          input_tensor=input_tensor)

#         for i, layer in enumerate(base_model_resnet50.layers):
#             # we must skip input layer, which has no weights
#             if i == 0:
#                 continue
#             base_model.layers[i+1].set_weights(layer.get_weights())

#         # add a global spatial average pooling layer
#         top_model = base_model.output
#         top_model = GlobalAveragePooling2D()(top_model)

#         # let's add a fully-connected layer
#         top_model = Dense(2048, activation='relu')(top_model)
#         top_model = Dense(2048, activation='relu')(top_model)
#         # and a logistic layer
#         predictions = Dense(numclasses, activation='sigmoid')(top_model)

#         # this is the model we will train
#         model = Model(inputs=base_model.input, outputs=predictions)
#     #     model = model.layers[-1].bias.assign([0.0]) # WIP getting an error ValueError: Cannot assign to variable dense_8/bias:0 due to variable shape (10,) and value shape (1,) are incompatible

#     #     model.summary()
#         last_chkpt_path = lcl_chkpt_dir + 'last_chkpt.h5'
#         if os.path.exists(last_chkpt_path):
#             print('New spot instance; loading previous checkpoint')
#             model.load_weights(last_chkpt_path)
#         elif starting_checkpoint:
#             print('No previous checkpoint found in opt/ml/checkpoints directory; loading checkpoint from', starting_checkpoint)
#             s3 = boto3.client('s3')
#             chkpt_name = lcl_chkpt_dir + '/' + 'start_chkpt.h5'
#             s3.download_file('canopy-production-ml-output', starting_checkpoint, chkpt_name)
#             model.load_weights(chkpt_name)
#         else:
#             print('No previous checkpoint found in opt/ml/checkpoints directory; start training from scratch')
    
#         return model
    
# #     random_id = random.randint(1,10001)

#     s3_chkpt_dir = s3_chkpt_base_dir + "/" + job_name   
#     base_name_checkpoint = "model_resnet"
#     save_checkpoint_s3 = SaveCheckpoints(base_name_checkpoint,lcl_chkpt_dir,s3_chkpt_dir)


#     # model_checkpoint_callback_loss = tf.keras.callbacks.ModelCheckpoint(
#     #   filepath= f'checkpoint/checkpoint_loss_{random_id}.h5',
#     #   format='h5',
#     #   verbose=1,
#     #   save_weights_only=True,
#     #   monitor='val_loss',
#     #   mode='min',
#     #   save_best_only=True)

#     # model_checkpoint_callback_recall = tf.keras.callbacks.ModelCheckpoint(
#     #   filepath= f'checkpoint/checkpoint_recall_{random_id}.h5',
#     #   format='h5',
#     #   verbose=1,
#     #   save_weights_only=True,
#     #   monitor='val_recall',
#     #   mode='max',
#     #   save_best_only=True)

#     # model_checkpoint_callback_precision = tf.keras.callbacks.ModelCheckpoint(
#     #   filepath= f'checkpoint/checkpoint_precision_{random_id}.h5',
#     #   format='h5',
#     #   verbose=1,
#     #   save_weights_only=True,
#     #   monitor='val_precision',
#     #   mode='max',
#     #   save_best_only=True)

#     # reducelronplateau = tf.keras.callbacks.ReduceLROnPlateau(
#     #   monitor='val_loss', factor=0.1, patience=5, verbose=1,
#     #   mode='min', min_lr=0.000001)

#     early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_precision', mode='max', patience=20, verbose=1)

#     wandb_callback = WandbCallback()
#     callbacks_list = [save_checkpoint_s3,early_stop, wandb_callback] #TODO re add reducelronplateau

#     model = define_model(config.numclasses, config.input_shape, starting_checkpoint, lcl_chkpt_dir)

#     # loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # Computes the cross-entropy loss between true labels and predicted labels.
#     # Focal loss instead of class weights: https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
#     model.compile(loss=SigmoidFocalCrossEntropy(), # https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
#                   optimizer=keras.optimizers.Adam(config.learning_rate),
#                   metrics=[tf.metrics.BinaryAccuracy(name='accuracy'), 
#                            tf.keras.metrics.Precision(name='precision'), # Computes the precision of the predictions with respect to the labels.
#                            tf.keras.metrics.Recall(name='recall'), # Computes the recall of the predictions with respect to the labels.
#                            F1Score(num_classes=10, name="f1_score") # https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
#                            ]
#     #               sample_weight_mode="temporal" # This argument is not supported when x is a dataset or a dataset iterator, instead pass sample weights as the third element of x.
#                  )
    
#     epochs = epochs
#     history = model.fit(gen.training_dataset, validation_data=gen.validation_dataset, 
#                         epochs=epochs, 
#                         callbacks=callbacks_list,
#                         shuffle=True # whether to shuffle the training data before each epoch
#                        ) 