FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.4.1-gpu-py37-cu110-ubuntu18.04

ENV PATH="/opt/ml/code:${PATH}"

# Install rasterio for reading TIFF files in TF Image Loader
RUN pip3 install rasterio geopandas shapely tensorflow tensorflow-addons[tensorflow]


# Copies the training code inside the container
COPY cb_feature_train4_aws.py /opt/ml/code/train.py
COPY data_loader_2.py /opt/ml/code/data_loader.py

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py