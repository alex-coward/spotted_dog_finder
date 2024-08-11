# Data Extraction Container

The data extraction container is part of the data pipeline used to extract, transform, and preprocess our project's image and annotation data for use in model training.

## Data Extraction

The `data_extraction/cli.py` file:
* Downloads and extracts images and annotations for the Stanford and Tsinghua dog breed datasets
* Downloads and extracts images and annotations for the Austin Pets Alive dataset
* Stores the images on GCS using GCSFuse and dask/multiprocessing

## Run Container in Vertex AI
This container can be run from the `workflow` container using the `--data_extraction` flag.

## To run from Container in a VM:
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
> `sudo docker run --rm -ti --privileged oll583921/spotted-data-extraction`

The container can also be run by installing Git and cloning the repo to your VM instance. Then, navigate to `src/data_extraction`, build and run the Docker image by executing `sh docker-shell.sh`.

#### Data Extraction

`data-extraction/cli.py` downloads datasets from their respective sources, extracts the archived file, and uploads the contents to a GCS bucket  

> `python cli.py [-a] [-b] [-i]`  
>> `-a   Download all datasets to GCS`  
>> `-b   Download all dog breed datasets to GCS`  
>> `-i   Download all individual dog datasets to GCS`

## Running the Data Pipeline

From the Docker pipenv shell, we can manage dog breed image data from downloading it from the source all the way through processing for model use. All GCS buckets are created as needed programmatically, so no other prep work is required.  The details are in the workflow/README.MD
