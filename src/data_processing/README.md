# Data Processing Container

The data processing container is part of the data pipeline used to extract, transform, and preprocess our project's image and annotation data for use in model training.

## Data Processing

The `data_processing/cli.py` file:
* Takes the data from a preprocessed bucket and converts the data into TFRecord files
* Creates TFRecords for both breed (cropped and uncropped) and individual dogs data. 
* Uses multiprocessing to improve performance.

## Run Container in Vertex AI
This container can be run from the `workflow` container using the `--data_processing` flag and a set of input arguments

>> `--image_size			Set the final image size for the dataset (default=224)`  
>> `--uncropped       		Use original breed images to produce dataset`   
>> `--cropped   			Use cropped breed images (from bounding boxes) to create dataset`  

## To run from Container in a VM:
* Create a service account with permissions for writing to cloud buckets (`roles/storage.objectAdmin`)
* Create a VM Instance from [GCP](https://console.cloud.google.com/compute/instances) 
* Under "Identity and API Access" attach your service account
* The data pipeline uses multi-threading, so it is recommened to select an instance with at least 8 CPU cores
* SSH into the newly created instance

## Starting the Container

Download and install Docker:
> `curl -fsSL https://get.docker.com -o get-docker.sh`  
> `sudo sh get-docker.sh`  

Run the Docker container from Docker Hub:  
> `sudo docker run --rm -ti --privileged oll583921/spotted-data-processing`

The container can also be run by installing Git and cloning the repo to your VM instance. Then, navigate to `src/data_processing`, build and run the Docker image by executing `sh docker-shell.sh`.

## Running the Data Pipeline

From the Docker pipenv shell, we can manage dog breed image data from downloading it from the source all the way through processing for model use. All GCS buckets are created as needed programmatically, so no other prep work is required. The details are in the workflow/README.MD
