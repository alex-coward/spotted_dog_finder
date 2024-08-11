# Full Dog tfRecords Dataset Serverless Model Training

This code is based off of the code provided for in the Mushroom App Serverless Training tutorial, with code modified for our dataset and models. The code allows serverless training jobs using Vertex AI:

## Prerequisites
* Have Docker installed
* Have WandB account
* Check that your Docker is running with the following command

`docker run hello-world`

### Install Docker 
Install `Docker Desktop`

### API's to enable in GCP for Project
Search for each of these in the GCP search bar and click enable to enable these API's
* Vertex AI API

### Setup GPU Quotas
In order to do serverless training we need access to GPUs from GCP.
- Go to [Quotas](https://console.cloud.google.com/iam-admin/quotas) in your GCP console
- Filter by `Quota: Custom model training` and select a GPU type, e.g: `Custom model training Nvidia T4 GPUs per region`

- Select a few regions
- Click on `EDIT QUOTAS`
- Put in a new limit and click `NEXT`
- Enter your Name and `SUBMIT REQUEST`
- This processes usually takes a few hours to get approved
- Also based on how new your GCP account is, you may not be approved


### Setup GCP Credentials
Next step is to enable our container to have access to Storage buckets & Vertex AI(AI Platform) in  GCP. 

#### Create a local **secrets** folder

It is important to note that we do not want any secure information in Git, so the secrets folder, which must be at the same level as the serverless_training folder, should be included in a .gitignore file.

#### Setup GCP Service Account
- Here are the step to create a service account:
- To setup a service account you will need to go to [GCP Console](https://console.cloud.google.com/home/dashboard), search for  "Service accounts" from the top search box. or go to: "IAM & Admins" > "Service accounts" from the top-left menu and create a new service account called "model-trainer". For "Service account permissions" select "Storage Admin", "AI Platform Admin", "Vertex AI Administrator".
- This will create a service account
- On the right "Actions" column click the vertical ... and select "Manage keys". A prompt for Create private key for "model-trainer" will appear select "JSON" and click create. This will download a Private key json file to your computer. Copy this json file into the **secrets** folder. Rename the json file to `model-trainer.json`

### GCS Bucket

- This code uses an existing GCP bucket, spotted/dog-breeds-trainer to upload files needed for serverless training. See the changes to code section below to how to change the bucket used.


### Get WandB Account API Key

We want to track our model training runs using WandB. Get the API Key for WandB: 
- Login into [WandB](https://wandb.ai/home)
- Go to to [User settings](https://wandb.ai/settings)
- Scroll down to the `API keys` sections 
- Copy the key
- Set an environment variable using your terminal: `export WANDB_KEY=...`

### Make any desired changes to code

-package/trainer/task.py & package/trainer/task_multi_gpu.py
 - These python files contain the code used to run serverless training on a single GPU/multi-GPU respectively. They will be modified by cli.sh or cli-multi-gpu.sh respectively and then uploaded to GCP by package-trainer.sh to be used by Vertex AI.
 - **DATASET:** They are currently set up to use the full dog breeds tfrecords located at gs://dog-breeds-224/processed/. To use different data, changes would need to be made to the data section, as indicated in files.
 - **MODELS:** They are currently set up to use "EfficientNetV2" and "EfficientNetV2_finetuning" as possible models. Make changes as indicated in files to incorporate other models. 
 - **FINETUNING:** The finetuning model is currently set up to continue running from a model-best.h5 file from a WandB run. Make changes as indicated in file to set which WandB run you would like to continue training, setting relevant training parameters. 
 - **WandB RUN:** Change run info as desired/needed.

-cli.sh & cli-multi-gpu.sh
 - These files are used to set evironmenrtal variables and command aguments that are used to determine various elements of training, including task.py, task_multi_gpu.py and cli.py 
 - export DISPLAY_NAME="dog_breed_training_multi_gpu_job_$UUID" [job name in GCP]
 - export MACHINE_TYPE="n1-standard-4" [type of mahcine used for job]
 - export REPLICA_COUNT=1 [number of workers]
 - export PYTHON_PACKAGE_URI=$GCS_BUCKET_URI/dog-breed-trainer.tar.gz [location of trainer file uploaded to GCP bucket] 
 - export ACCELERATOR_TYPE="NVIDIA_TESLA_T4" [type of GPU]
 - export ACCELERATOR_COUNT=1 [number of GPUs]
 - export GCP_REGION="us-central1" # Adjust region based on you approved quotas for GPUs
 - export CMDARGS="--model_name=EfficientNetV2,--epochs=30,--batch_size=32,--wandb_key=$WANDB_KEY" [comment out all but one of these to set which model and model training parameters will be used]

-cli.py 
 -  DISPLAY_NAME = "dog_breeds_" + job_id
 - python_package_gcs_uri=f"{GCS_BUCKET_URI}/dog-breed-trainer.tar.gz"
 - print(f"{GCS_BUCKET_URI}/dog-breed-trainer.tar.gz")

-setup.py
 - Change name and description from current related to dog-breeds-trainer if desired/relevant

-docker-shell.sh
 - Builds docker container
 - If not using M1/M2 mac, be sure to remove comment from  "#docker build -t $IMAGE_NAME -f Dockerfile ." and to add a comment to  "docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile ."
 - export GCS_BUCKET_URI="gs://dog-breeds-trainer" [sets bucket used by package-trainer.sh to upload training files]
 - export GCP_PROJECT="spotted-399806 " [sets project used] 

-package-trainer.sh
 - builds and uploads files in trainer directory, modified by cli.sh or cli-multi-gpu.sh
 - change gsutil cp trainer.tar.gz $GCS_BUCKET_URI/dog-breed-trainer.tar.gz to match desired file name, matching the name used in cli.sh or cli-multi-gpu.sh

## Run Container

### Run `docker-shell.sh`
Only built for Mac (docker-shell.sh). If anyone is on a windows machine and needs to run docker-shell.bat we can adjust that file as well.
```
- Make sure you are inside the `model_training` folder and open a terminal at this location
- Run `sh docker-shell.sh` 
- The `docker-shell` file assumes you have the `WANDB_KEY` as an environment variable and is passed into the container

### Package & Upload Python Code

### Run `sh package-trainer.sh`
- This script will create a `trainer.tar.gz` file with all the training code bundled inside it
- Then this script will upload this packaged file to your GCS bucket, name currently set as `dog-breeds-trainer.tar.gz`

### Create Jobs in Vertex AI
- Run `sh cli.sh`
- `cli.sh` is a script file to make calling `gcloud ai custom-jobs create` easier by maintaining all the parameters in the script

### OPTIONAL:Create Jobs in Vertex AI using CPU
- Edit your `cli.sh` to not pass the `accelerator-type` and `accelerator-count`
- Run `sh cli.sh`

### View Jobs in Vertex AI
- Go to Vertex AI [Custom Jobs](https://console.cloud.google.com/vertex-ai/training/custom-jobs)
- You will see the newly created job ready to be provisioned to run. 

### View Training Metrics
- Go to [WandB](https://wandb.a)
- Select the project `dog-breeds-serverless-training` (or other name as modifed per the instructions above)
- You will view the training metrics tracked and automatically updated

### OPTIONAL: Multi GPU Training
- Change `cli-multi-gpu.sh` to use the number of GPUs you want
- Run `sh cli-multi-gpu.sh`
- `cli-multi-gpu.sh` is a script file to make calling `gcloud ai custom-jobs create` easier by maintaining all the parameters in the script