# Project Canopy - Congo Basin Feature Detection Documentation

Each of the headings below refer to a subdirectory representing a discrete phase of this project 


## model-development
 <br />
Description: all model training and model validation resources
<br />
<br />

Requirements:

* IAM Role under Project Canopy AWS Account
* Full Sagemaker priveledges 
* Full EFS priveledges 
* EFS Volume of Bucket containing train/test/val chips from S3

Assets:

* All previously generated checkpoint files: s3://canopy-production-ml-output/ckpt/


* Training chips acquired via GEE utilizing PC's custom cloud masking preprocessing solution
  * Most current version of chips (per label) utilized by label lists in the same directory 
    * s3://canopy-production-ml/chips/cloudfree-merge-polygons/dataset_v2/
  * Original chips (per label)
    * s3://canopy-production-ml/chips/cloudfree-merge-polygons/yes/  
  * All chips (unsorted) acquired via GEE utilizing Sentinal Hub's S2Cloudless cloud-free dataset
    * s3://canopy-production-ml/chips/s2cloudless/


* training / test / val lists in CSV format containing relative file paths to all respective chip assets to be utilized for Tensorflow's Data Generator 
  * Parent directory within here:  s3://canopy-production-ml/chips/cloudfree-merge-polygons/dataset_v2/


* <b>new_labels.json</b> - contained within the git parent directory, this file maps integers to specific label names. Required for use <b>test_generator.py</b> script 

<br />

Notebooks:

* <b> Sagemaker_TensorFlow_Training </b> - For local testing and cloud training optimized for Sagemaker's TensorFlow Estimator. Utilizes <b>train_no_s3.py</b> script
* <b> evaluation_master </b> - For evaluating performance of specific checkpoint (model weights). Careful attention should be placed on ensuring you are using the same band combination as the training session associated with the same checkpoint. Evaluation notebook mostly utilized within a Sagemaker on-demand notebook environment due to relatively long feedback time when running evaluation on locally stored chips. Will require an EFS volume. Sample EFS volume mounting code contained at the top of the notebook.      

<br />

Scripts:

* <b> train_no_s3.py </b> - Used for the SageMaker TensorFlow Estimator for training. 
* <b> test_generator.py </b> - Used for packaging the respective evaluation chips dataset within the <b>evaluation_master</b> notebook.   



## inference

