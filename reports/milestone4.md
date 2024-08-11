# AC215 - Milestone 4 - SPOTTED!

**Team Members**  
Alex Coward, Olga Leushina, and Jonathan Sessa

**Group Name**  
Spotted!

**Project**  
Our project seeks to identify lost dogs from images uploaded by spotters and find potential matches among dogs reported missing by owners. To accomplish this we envision requiring two models: one to predict the breed of a dog from a given photo (to help group images and filter likely matches) and another to predict the likelihood of two images depicting the same dog.


### Milestone 4

Our entire data pipeline now runs serverless using Vertex AI. To accomodate this change, we refactored our entire data pipeline so each step now runs in its own container. We have also added a workflow container to manage the process, which includes the script for describing our Directed Acyclic Graph (DAG) - which specifies the required order for each container to run. The script allows us to run each step individually or all together on Vertex AI, and also allocates addition CPUs for each step to maximize efficiency and use of multiprocessing for data processing.


#### Code Structure

**Data Storeage**

All of our data is now stored and accessed via GCS buckets. The `data` folder on our repo does not contain any data, just information (e.g., CSV files, screenshots, etc.) and links to data sources.


**Data Processing Containers**

Our data processing containers are responsible for managing each step of our data pipeline, from downloading the images and annotations from the source datasets, transforming and processing the images, to ultimately creating the TFRecord files for our models to use for training.

Each step of our data pipeline is stored in its own container and can be run individually and serverless on Vertex AI using our workflow container (see below):

* `src/data_extraction` - Downloads datasets (Stanford, Tsinghua, Austin Pets) from source, extracts files and annotations, and stores files on GCS.

* `src/data_transformation` - Extracts label data and renames files, then moves images to a pooled GCS bucket for all dog breed data. Also produces cropped versions of dog breed images using bounding boxes from annotations. Uses multiprocessing to improve performance.

* `src/data_preprocessing` - Generates our train/validation/test splits (stratifying on breed to ensure proportional representation), generates CSV files, and resizes images based on input parameter. Creates datasets for cropped and uncropped breed data. Uses multiprocessing to improve performance.

* `src/data_processing` - Takes the data from a preprocessed bucket and converts the data into TFRecord files. Creates TFRecords for both breed (cropped and uncropped) and individual dogs data. Uses multiprocessing to improve performance.


**Model Training Container**

* `src/model_training` 
* `src/serverless_training_distillation`

Model trainer container starts Vertex AI training job defined by the input parameter when running cli.sh or cli-multi-gpu.sh

Serverless_training_distillation container has the same functionality and internal structure as model_training container. It can be run in the workflow as an alternative. 

The code inside additionally allowes for running distillation over the pre-trained models by setting 
cli.sh file with params like
export CMDARGS="--model_name=Distilled_EfficientNetV2_finetune,--epochs=30,--batch_size=32,--wandb_key=$WANDB_KEY" 

Please note that in both containers in task.py and task_multi-gpu.py inside the /trainer folder there are direct references to pre-trained models on Wandb that have to be replaced before the distilled models are built.

Our serverless training report with distilled model can be found here:
https://api.wandb.ai/links/spotted-dog/cfpuf217 

**Workflow Orchestration Container**

As mentioned above, our entire workflow now runs serverless in Vertex AI. Each component of the data pipeline can be run invidually or as a whole (end-to-end) from `src/workflow/cli.py`. In addition, the `model_training` container can also be called:

> `python cli.py`  
>> `--pipeline              Run all data pipeline containers in the proper order`  
>> `--data_extraction       Run only the data_extraction container`   
>> `--data_transformation   Run only the data_transformation container`  
>> `--data_preprocessing    Run only the data_preprocessing container`  
>> `--data_processing       Run only the data_processing container`  
>> `--model_training        Run only the model_training container`  

Additional flags can also be used to customize the final output of the data pipeline:

> `python cli.py --pipeline`  
>> `--image_size			Set the final image size for the dataset (default=224)`  
>> `--uncropped       		Use original breed images to produce dataset`   
>> `--cropped   			Use cropped breed images (from bounding boxes) to create dataset`  
>> `--individual    		Use individual dog images to create dataset`  

For example, to create a dataset for training our breed classifier using cropped breed images:  

> `python cli.py --pipeline --image_size 224 --cropped`  

Or to create a dataset for training our object detection model using the original breed images and bounding box annotations:  

> `python cli.py --pipeline --image_size 256 --uncropped`  

![pipeline](./reports/screenshots/vertex-ai-pipeline.png)


**Data Versioning (src/data_versioning)**

The `data_versioning` container stores files for configuring our Docker container for running DVC and storing necessary DVC files (instructions for running Dockerfile can be found [here](../src/data_versioning/README.md)). All DVC data files are stored in a `dvc_store` folder in the GCS data bucket.



**Notebooks** 
Notebooks used for data exploration and initial model training and exploration:

* `eda.ipynb` - Exploratory Data Analysis of our combined dog breed datasets

* `classifier_with_tfr.ipynb` - Code for accessing our TFRecords files in GCS to parse and prepare for model training.

* `object_detection.ipynb` - Notebook for our initial test of fine-tuning an object detection model using our data.