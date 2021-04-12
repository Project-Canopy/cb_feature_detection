import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.losses import CategoricalCrossentropy

# tf.keras.losses.CategoricalCrossentropy

import numpy as np
import argparse, os, subprocess, sys
import tensorflow.keras as keras
import pandas as pd
import io
import json
import random
import h5py
import multiprocessing
import time

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Script mode doesn't support requirements.txt
# Here's the workaround:
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


class DataLoader:
    def __init__(self,
                 training_dir="./",
                 label_file_path_train="labels_test_v1.csv",
                 label_file_path_val="labels_val.csv",
                 bucket_name='canopy-production-ml',
                 data_extension_type='.tif',
                 bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                 augment=False,
                 enable_shuffle=False,
                 training_data_shuffle_buffer_size=1000,
                 training_data_batch_size=20,
                 enable_data_prefetch=False,
                 data_prefetch_size=tf.data.experimental.AUTOTUNE,
                 num_parallel_calls=tf.data.AUTOTUNE,
                 output_shape=(tf.float32, tf.float32)):

        self.bucket_name = bucket_name

        self.label_file_path_train = label_file_path_train
        self.label_file_path_val = label_file_path_val

        print(f"label_file_path_train: {self.label_file_path_train}")
        print(f"labels_file_val: {self.label_file_path_val}")
        self.labels_file_train = pd.read_csv(self.label_file_path_train)
        self.labels_file_val = pd.read_csv(self.label_file_path_val)
        self.training_filenames = self.labels_file_train.paths.to_list()

        self.labels_file_val = pd.read_csv(self.label_file_path_val)
        self.validation_filenames = self.labels_file_val.paths.to_list()

        self.bands = bands
        self.augment = augment
        self.local_path_train = training_dir

        self.enable_shuffle = enable_shuffle
        if training_data_shuffle_buffer_size is not None:
            self.training_data_shuffle_buffer_size = training_data_shuffle_buffer_size

        self.num_parallel_calls = num_parallel_calls
        self.enable_data_prefetch = enable_data_prefetch
        self.data_prefetch_size = data_prefetch_size
        self.data_extension_type = data_extension_type
        self.output_shape = output_shape

        self.training_data_batch_size = training_data_batch_size

        self.build_training_dataset()
        self.build_validation_dataset()

    def build_training_dataset(self):
        # Tensor of all the paths to the images
        self.training_dataset = tf.data.Dataset.from_tensor_slices(self.training_filenames)

        # If data augmentation
        if self.augment is True:
            # https://stackoverflow.com/questions/61760235/data-augmentation-on-tf-dataset-dataset
            print("Data augmentation enabled")
            
            self.training_dataset = self.training_dataset.map((
                lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
                num_parallel_calls=self.num_parallel_calls).map(
            lambda image, label: (tf.image.random_flip_left_right(image), label)
            ).map(
            lambda image, label: (tf.image.random_flip_up_down(image), label)
            ).repeat(3)
            
            self.length_training_dataset = len(self.training_filenames) * 3
            print(f"Training on {self.length_training_dataset} images")
        else:
            print("No data augmentation. Please set augment to True if you want to augment training dataset")
            self.training_dataset = self.training_dataset.map((
                lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
                num_parallel_calls=self.num_parallel_calls)
            self.length_training_dataset = len(self.training_filenames)
            print(f"Training on {self.length_training_dataset} images")

        # Randomly shuffles the elements of this dataset.
        # This dataset fills a buffer with `buffer_size` elements, then randomly
        # samples elements from this buffer, replacing the selected elements with new
        # elements. For perfect shuffling, a buffer size greater than or equal to the
        # full size of the dataset is required.
        if self.enable_shuffle is True:
            if self.training_data_shuffle_buffer_size is None:
                self.training_data_shuffle_buffer_size = len(self.length_training_dataset)
            self.training_dataset = self.training_dataset.shuffle(self.training_data_shuffle_buffer_size,
                                                                  reshuffle_each_iteration=True
                                                                  # controls whether the shuffle order should be different for each epoch
                                                                  )

        if self.training_data_batch_size is not None:
            # Combines consecutive elements of this dataset into batches
            self.training_dataset = self.training_dataset.batch(self.training_data_batch_size)

        # Most dataset input pipelines should end with a call to `prefetch`. This
        # allows later elements to be prepared while the current element is being
        # processed. This often improves latency and throughput, at the cost of
        # using additional memory to store prefetched elements.
        if self.enable_data_prefetch:
            self.training_dataset = self.training_dataset.prefetch(self.data_prefetch_size)

    def build_validation_dataset(self):
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(list(self.validation_filenames))
        self.validation_dataset = self.validation_dataset.map(
            (lambda x: tf.py_function(self.process_path, [x], self.output_shape)),
            num_parallel_calls=self.num_parallel_calls)

        self.validation_dataset = self.validation_dataset.batch(self.training_data_batch_size)
        print(f"Validation on {len(self.validation_filenames)} images ")

#     def read_image(self, path_img):
#         path_to_img = self.local_path_train + "/" + path_img.numpy().decode()
#         train_img = np.transpose(rasterio.open(path_to_img).read(self.bands), (1, 2, 0))
#         # Normalize image
#         train_img = tf.image.convert_image_dtype(train_img, tf.float32)
#         return train_img
    
    def read_image(self, path_img):
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
        # testing if path in the training csv file or in the val one
        if path_img.numpy().decode() in self.labels_file_train.paths.to_list():
            ### Training csv
            # path_img is a tf.string and needs to be converted into a string using .numpy().decode()
            id = int(self.labels_file_train[self.labels_file_train.paths == path_img.numpy().decode()].index.values[0])
            # The list of labels (e.g [0,1,0,0,0,0,0,0,0,0] is grabbed from the csv file on the row where the s3 path is
            label = self.labels_file_train.drop('paths', 1).iloc[int(id)].to_list()
        else:
            ### Validation csv
            # path_img is a tf.string and needs to be converted into a string using .numpy().decode()
            id = int(self.labels_file_val[self.labels_file_val.paths == path_img.numpy().decode()].index.values)
            # The list of labels (e.g [0,1,0,0,0,0,0,0,0,0] is grabbed from the csv file on the row where the s3 path is
            label = self.labels_file_val.drop('paths', 1).iloc[int(id)].to_list()
        return label

    # Function used in the map() and returns the image and label corresponding to the file_path input
    def process_path(self, file_path):
        label = self.get_label_from_csv(file_path)
        img = self.read_image(file_path)
        return img, label


class SaveCheckpoints(keras.callbacks.Callback):
    def __init__(self, 
                base_name_checkpoint=None,
                lcl_chkpt_dir=None,
                s3_chkpt_dir=None):

        self.base_name_checkpoint = base_name_checkpoint
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
    print("TensorFlow version", tf.__version__)
    print("Keras version", keras.__version__)
    # Keras-metrics brings additional metrics: precision, recall, f1
    install('keras-metrics')
    import keras_metrics

    install('tensorflow-addons')
    from tensorflow_addons.metrics import F1Score, HammingLoss
    from tensorflow_addons.losses import SigmoidFocalCrossEntropy
    from tensorflow_addons.optimizers import CyclicalLearningRate

    install('wandb')
    import wandb
    from wandb.keras import WandbCallback

    install('rasterio')
    import rasterio
    from rasterio.session import AWSSession

    install('boto3')
    import boto3

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--augment', type=str, default="False")
    parser.add_argument('--flip_left_right', type=str, default="False")
    parser.add_argument('--flip_up_down', type=str, default="False")
    parser.add_argument('--rot90', type=str, default="False")
    parser.add_argument('--numclasses', type=float, default=10)
    parser.add_argument('--bands', required=True)
    parser.add_argument('--bucket', type=str, default="margaux-bucket-us-east-1")
    parser.add_argument('--training_file', type=str, default="labels_test_v1.csv")
    parser.add_argument('--validation_file', type=str, default="val_labels.csv")
    parser.add_argument('--wandb_key', type=str, default=None)
    # data directories
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    parser.add_argument('--s3_chkpt_dir', type=str)
    parser.add_argument('--output', type=str, default="/opt/ml/output")
    parser.add_argument('--job_name', 
                        type=str, default=json.loads(os.environ.get('SM_TRAINING_ENV'))["job_name"])
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--starting_checkpoint', type=str, default=None)

    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args, _ = parser.parse_known_args()

    os.environ["WANDB_API_KEY"] = args.wandb_key   # "6607ed7a49b452c2f3494ce60f9514f6c9e3b4e6"
    wandb.init(project='project-canopy')  # , sync_tensorboard=True
#     wandb.init(project='project-canopy')  # , sync_tensorboard=True
    config = wandb.config
    
    job_name = args.job_name
    data_dir = args.data
    lcl_chkpt_dir = "/opt/ml/model/checkpoints"
    
    s3_chkpt_base_dir = args.s3_chkpt_dir
    starting_checkpoint = args.starting_checkpoint
    print('Starting checkpoint:', starting_checkpoint)
    
    if not os.path.exists(lcl_chkpt_dir):
        os.mkdir(lcl_chkpt_dir)

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    bucket = args.bucket
    training_file = args.training_file
    validation_file = args.validation_file

    # Had to use strings because of the way sagemaker script was passing the args values 
    if args.augment.lower() == "false":
        augmentation_data = False
    else:
        augmentation_data = True

    print(f"lr: {lr}, batch_size: {batch_size}, augmentation_data: {augmentation_data}")
    
    bands = args.bands.split(" ")
    bands = list(map(int, bands))   # convert to int
    print(f"bands {bands}")
    input_shape = (100, 100, int(len(bands)))
    print(f"Input shape: {input_shape}")

    numclasses = int(args.numclasses)

    gpu_count = args.gpu_count
    model_dir = args.model_dir
    training_dir = args.training
    
    print(os.system(f"ls {training_dir}"))
    
    
    # validation_dir = args.validation

    def define_model(numclasses, input_shape, starting_checkpoint=None, lcl_chkpt_dir=None):
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
            base_model.layers[i + 1].set_weights(layer.get_weights())

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
#         model.summary()
        last_chkpt_path = lcl_chkpt_dir + 'last_chkpt.h5'
        if os.path.exists(last_chkpt_path):
            print('New spot instance; loading previous checkpoint')
            model.load_weights(last_chkpt_path)
        elif starting_checkpoint:
            print('No previous checkpoint found in opt/ml/checkpoints directory; loading checkpoint from', starting_checkpoint)
            s3 = boto3.client('s3')
            chkpt_name = lcl_chkpt_dir + '/' + 'start_chkpt.h5'
            s3.download_file('canopy-production-ml-output', starting_checkpoint, chkpt_name)
            model.load_weights(chkpt_name)
        else:
            print('No previous checkpoint found in opt/ml/checkpoints directory; start training from scratch')
    
        return model
    
    def define_model_EfficientNetB0(numclasses, input_shape, starting_checkpoint=None, lcl_chkpt_dir=None):
        # parameters for CNN
        input_tensor = Input(shape=input_shape)

        # introduce a additional layer to get from bands to 3 input channels
        input_tensor = Conv2D(3, (1, 1))(input_tensor)

        base_model_resnet50 = keras.applications.EfficientNetB0(include_top=False,
                                                          weights="imagenet",
                                                          input_shape=(100, 100, 3))
        
        base_model = keras.applications.EfficientNetB0(include_top=False,
                                                 weights=None,
                                                 input_tensor=input_tensor)

        for i, layer in enumerate(base_model_resnet50.layers):
            # we must skip input layer, which has no weights
            if i == 0:
                continue
            if args.freeze_bn_layer.lower() == "true":
                if "bn" in layer.name:
                    layer.trainable = False
                
            base_model.layers[i + 1].set_weights(layer.get_weights())

        # add a global spatial average pooling layer
        top_model = base_model.output
        top_model = GlobalAveragePooling2D()(top_model)

        # let's add a fully-connected layer
        top_model = Dense(2048, activation='relu')(top_model)
        top_model = Dense(2048, activation='relu')(top_model)
        # and a logistic layer  
        predictions = Dense(numclasses, activation="sigmoid")(top_model)
        

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
#         model.summary()
        last_chkpt_path = lcl_chkpt_dir + 'last_chkpt.h5'
        if os.path.exists(last_chkpt_path):
            print('New spot instance; loading previous checkpoint')
            model.load_weights(last_chkpt_path)
        elif starting_checkpoint:
            print('No previous checkpoint found in opt/ml/checkpoints directory; loading checkpoint from', starting_checkpoint)
            s3 = boto3.client('s3')
            chkpt_name = lcl_chkpt_dir + '/' + 'start_chkpt.h5'
            s3.download_file('canopy-production-ml-output', starting_checkpoint, chkpt_name)
            model.load_weights(chkpt_name)
        else:
            print('No previous checkpoint found in opt/ml/checkpoints directory; start training from scratch')
    
        return model
    

    s3_chkpt_dir = s3_chkpt_base_dir + "/" + job_name
    base_name_checkpoint = "model_resnet"
    save_checkpoint_s3 = SaveCheckpoints(base_name_checkpoint, lcl_chkpt_dir, s3_chkpt_dir)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_recall', mode='max', patience=20, verbose=1)
    

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',mode='min', factor=0.1,patience=5, min_lr=0.00001, verbose=1)
    
    clr = CyclicalLearningRate(initial_learning_rate=.00001,
                                        maximal_learning_rate=.0001,
                                        step_size=5000, 
                                        scale_fn=lambda x: 1 / (2.0 ** (x - 1)), 
                                        scale_mode='cycle',
                                        name='CyclicalLearningRate')
    
    lrs = tf.keras.callbacks.LearningRateScheduler(clr, verbose=1)

    callbacks_list = [save_checkpoint_s3, early_stop,reduce_lr, WandbCallback()]

    ######## WIP: multi GPUs ###########
    # if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    #     print("Running on GPU using mirrored_strategy")
    #     list_gpus = []
    #     for gpu in range(0, gpu_count):
    #         list_gpus.append(f"/gpu:{gpu}")
    #
    #     print(f"list_gpus: {list_gpus}")
    #     # https://towardsdatascience.com/quick-start-to-multi-gpu-deep-learning-on-aws-sagemaker-using-tf-distribute-9ee08bc9612b
    #     mirrored_strategy = tf.distribute.MirroredStrategy(devices=list_gpus)
    #     batch_size = batch_size * mirrored_strategy.num_replicas_in_sync
    #     with mirrored_strategy.scope():
    #         model = define_model(numclasses, input_shape, starting_checkpoint, lcl_chkpt_dir)
    #     with mirrored_strategy.scope():
    #       # Set reduction to `none` so we can do the reduction afterwards and divide by
    #       # global batch size.
    #       loss_object = SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.NONE)
    #       def compute_loss(labels, predictions):
    #         per_example_loss = loss_object(labels, predictions)
    #         return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    #         model.compile(loss=SigmoidFocalCrossEntropy(),
    #                       # https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
    #                       optimizer=keras.optimizers.Adam(lr),
    #                       metrics=[tf.metrics.BinaryAccuracy(name='accuracy'),
    #                                tf.keras.metrics.Precision(name='precision'),
    #                                # Computes the precision of the predictions with respect to the labels.
    #                                tf.keras.metrics.Recall(name='recall'),
    #                                # Computes the recall of the predictions with respect to the labels.
    #                                F1Score(num_classes=numclasses, name="f1_score")
    #                                # https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
    #                                ]
    #                       )
    # else:
    #     print("Running on CPU")
    model = define_model(numclasses, input_shape, starting_checkpoint, lcl_chkpt_dir)

#     model = define_model_EfficientNetB0(numclasses, input_shape,starting_checkpoint, lcl_chkpt_dir)


    model.compile(loss=BinaryCrossentropy(),
                  # https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy
                  optimizer=keras.optimizers.Adam(lr),
                  metrics=[tf.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           # Computes the precision of the predictions with respect to the labels.
                           tf.keras.metrics.Recall(name='recall'),
                           # Computes the recall of the predictions with respect to the labels.
                           F1Score(num_classes=numclasses, name="f1_score")
                           # https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
                           ]
                  )

    print(f"training_dir {os.path.join(training_dir, training_file)}")
    print(f"val_dir {os.path.join(training_dir, validation_file)}")

    print("Dataloader initialization...")
    gen = DataLoader(training_dir=training_dir,
                     label_file_path_train=os.path.join(training_dir, training_file),
                     label_file_path_val=os.path.join(training_dir, validation_file),
                     bucket_name=bucket,
                     data_extension_type='.tif',
                     bands=bands,
                     augment=augmentation_data,
                     enable_shuffle=True,
                     training_data_batch_size=batch_size,
                     enable_data_prefetch=True)
    
    history = model.fit(gen.training_dataset,
                        validation_data=gen.validation_dataset,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        )
