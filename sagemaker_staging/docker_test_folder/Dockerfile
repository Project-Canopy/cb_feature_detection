FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.4.1-cpu-py37-ubuntu18.04

ENV PATH="/opt/ml/code:${PATH}"

# Install rasterio for reading TIFF files in TF Image Loader
RUN pip3 install rasterio keras

# Copies the training code and label files inside the container
COPY cb_feature_train1_aws.py /opt/ml/code/train.py

# COPY labels_test_v1.csv labels_test_v1.csv
# COPY val_labels.csv val_labels.csv
# COPY labels.json labels.json

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py