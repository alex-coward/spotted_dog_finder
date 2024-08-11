# Workflow Container

The workflow container can be run locally or on a GCP VM to initiate Vertex AI Pipelines for server-less data processing or model training.


## Running Vertex AI Pipelines

From the Docker pipenv shell, the following commands can be run each container separately, or the entire data pipeline.

The pipeline expects the following GCS buckets to be accessible from the GCS Bucket root:
* `spot-raw-data`
* `spot-breed-source`
* `spot-breed-data`
* `spot-individual-data`
* `spotted-pipelines1`


#### Full Data Pipeline

As mentioned above, our entire workflow now runs serverless in Vertex AI. Each component of the data pipeline can be run invidually or as a whole (end-to-end) from `src/workflow/cli.py`. To run the full pipeline run:

> `python cli.py --pipeline` 

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


#### Data Extraction

`src/data-extraction/cli.py` downloads datasets (Stanford, Tsinghua, Austin Pets) from source, extracts files and annotations, and stores files on GCS. 

> `python cli.py --data_extraction`  


#### Data Transformation

`src/data-transformation/cli.py` extracts label data and renames files, then moves images to a pooled GCS bucket for all dog breed data. Also produces cropped versions of dog breed images using bounding boxes from annotations. Uses multiprocessing to improve performance.

> `python cli.py --data_transformation`  


#### Data Preprocessing

`src/data-preprocessing/cli.py` generates our train/validation/test splits (stratifying on breed to ensure proportional representation), generates CSV files, and resizes images based on input parameter. Creates datasets for cropped and uncropped breed data. Uses multiprocessing to improve performance.

> `python cli.py --data-preprocessing`  
>> `--image_size			Set the final image size for the dataset (default=224)`  
>> `--uncropped       		Use original breed images to produce dataset`   
>> `--cropped   			Use cropped breed images (from bounding boxes) to create dataset`  


#### Data Processing

`src/data-processing/cli.py` - Takes the data from a preprocessed bucket and converts the data into TFRecord files. Creates TFRecords for both breed (cropped and uncropped) and individual dogs data. Uses multiprocessing to improve performance.

> `python cli.py --data-processing`  
>> `--image_size			Set the final image size for the dataset (default=224)`  
>> `--uncropped       		Use original breed images to produce dataset`   
>> `--cropped   			Use cropped breed images (from bounding boxes) to create dataset`  
>> `--individual   			Use individual dog images to create dataset`  


#### Model Training and deployment

In order to run Model training in full please follow instructions in src/model_training/README.md
In order to run Model deployment from WanDB please follow instructions in src/model_deployment/README.md

In order to run model training and deployment from the Workflow container (without initiating full pipeline) you can run the following commands:

`python cli.py --model-training` 

`python cli.py --model-deploy`


## Initial Setup of GCP VM - if running from the VM
* Create a service account with permissions for writing to cloud buckets (`roles/storage.objectAdmin`)
* Create a VM Instance from [GCP](https://console.cloud.google.com/compute/instances) 
* Set region to the same as GCS for improved performance (defined in `setup.sh`)
* Under "Identity and API Access" attach your service account
* The data pipeline uses multi-threading, so it is recommened to select an instance with at least 8 CPU cores
* SSH into the newly created instance


## Starting the Container

Download and install Docker:
> `curl -fsSL https://get.docker.com -o get-docker.sh`  
> `sudo sh get-docker.sh`  

Run the Docker container from Docker Hub:  
> `sudo docker run --rm -ti --privileged oll583921/spotted-data-pipeline-workflow`

The container can also be run by installing Git and cloning the repo to your VM instance. Then, navigate to `src/workflow`, build the Docker image using the Dockerfile and run the image.