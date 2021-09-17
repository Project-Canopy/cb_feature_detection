# Project Canopy - Congo Basin Feature Detection Documentation

Each of the headings below refer to a subdirectory representing a discrete phase of the project. The data collection process is documented with pipeline code in the <b>Cloudfree-Merging</b> Git repository.     

<br />

## <u>model-development</u>
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

<br />

## <u>inference</u>

<br />

Requirements:

* <b>Cortex</b> - Have sufficient priveleges to enable the cortex orchestration / scheduling tool to create clusters via CloudFormation, modifications of VPC's etc. API setup guide [here](https://docs.cortex.dev/).  
* Source datasets should have the 3257 EPSG to compatible with the trained model  

<br />

Notebooks:

* <b> Cortex_Post </b> - API Calls to Cortex utilizing an <b>incomplete_granules</b> txt file to complete the inference process
* <b> granules_convert_to_3257 </b> - Convert granules to 3257 prior to inference
* <b> inference_pipeline </b> - full code for prorotyping inference process. includes code for reading in inference output results prior to precossing steps for displaying inference results which is located in the <b>display</b> folder located in the parent directory in the repository.   

<br />

Scripts:

* <b> inference_predict_cortex.py </b> - Script to be used for running inference pipeline via Cortex 
* <b> inference_predict.py </b> - Script to be used for running inference pipeline. As opposed to the Cortex script, this script is generalizable to both local and cloud-based computing solutions. 

Cortex Config Files:

* <b>cluster_config.yml</b> - File for launching the KB cluster with all associated VPC settings and provisioned instances. See cortex [documentation](https://docs.cortex.dev) for details.  
* <b>cortex_batch.yml</b> - For running the specific workload within the cluster. Batch is a more viable solution for a workload with the size of our initial inference job. 
* <b>cortex_realtime.yml</b> - For running the specific workload within the cluster. Better for a short discrete workload. 


Assets:

* All necessary asset paths (inputs) are hardcoded within the generalized and cortex-specific scripts. They are also listed below in addition to output files from inference process. Parent dir - s3://canopy-production-ml/inference/  


* Geometry Predictions - s3://canopy-production-ml/inference/geometry_predictions/
  * The master ISL prediction output currently in use on Ellipsis Drive
    * CB_ISL_Full_Predict_05242021.geojson 
  * Sample of False Positive ISL chips for research
    * false_positive_ISL_labelled.geojson
  * Full Line geometry of source ISL labels used for training
    * ISL_Training_Labels.geojson  
  * OSM metadata within the Congo Basin potentially useful for performing JOINS against the full ISL predictions (inference) for further refinement of the inference results 
    * osm_relevant_clipped.geojson
  * Inference output INNER JOINED with the source ISL labels used for training 
    * true_positive_ISL_labelled.geojson


* The final model and weights for inference  
  * s3://canopy-production-ml/inference/model_files/

* Historic inference. Only major different between rounds is level of completeness given that the underlying code and model / weights inputs are identical. 
  * s3://canopy-production-ml/inference/output/

* Full Congo Basin pull (from Google Earth Engine) - separated by granules 
  * Utilizes Project Canopy's custom cloud masking preprocessing solution 
  * s3://canopy-production-ml/full_congo_basin/02.17.21_CB_GEE_Pull/


<b>Recommendation: Train and run inference on the S2Cloudless dataset and compare results with PC's dataset. May be very competitive and requires a lot less code!</b>

* Full Congo Baasin pull (from Google Earth Engine) 
  * Utilizes Sentinel Hub's S2Cloudless dataset available directly via Google Earth Engine
  * s3://canopy-production-ml/full_congo_basin/full_congo_s2cloudless_3/






