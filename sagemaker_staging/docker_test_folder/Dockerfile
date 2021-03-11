FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.4.1-gpu-py37-cu110-ubuntu18.04

ENV PATH="/opt/ml/code:${PATH}"

# Install rasterio for reading TIFF files in TF Image Loader
RUN pip3 install rasterio wandb tensorflow-addons

# Copies the training code inside the container
COPY cb_feature_train2_aws.py /opt/ml/code/train.py
 
# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py