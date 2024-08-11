# Data Transformation Container

The data transformation container is part of the data pipeline used to extract, transform, and preprocess our project's image and annotation data for use in model training.

## Data Transformation

The `src/data_transformation/cli.py` file:  
* Extracts breed label names  
* Converts the filenames for all breed data to a standard format  
* Extracts information from source annotations and creates new JSON versions  
* Crops images based on included bounding box information  

## Run Container in Vertex AI
This container can be run from the `workflow` container using the `--data_transformation` flag.

This container expects `spot-raw-data` bucket to be created with subfolders for `standford` and `tsinghua`  containing the raw data from those dataset; and requires a `spot-breed-source` bucket created for output.


## To Run from Container in a VM:
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

The container can also be run by installing Git and cloning the repo to your VM instance. Then, navigate to `src/data_transformation`, build and run the Docker image by executing `sh docker-shell.sh`.

## Running the Data Pipeline

From the Docker pipenv shell, we can manage dog breed image data from downloading it from the source all the way through processing for model use. All GCS buckets are created as needed programmatically, so no other prep work is required.  The details are in the workflow/README.MD