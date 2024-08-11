# Model deployment container
This container is used to download model from WanDB, deploy it to Vertex AI and run prediction using files in the /data folder.

## Prerequisites - Docker and VS code (as always)

## Setup Environments
**You will need your GCP account set up for all cases. In order to run --upload and --deploy (if --upload wasn't done) you will need also WandB account.**

### API's to enable in GCP for Project
Search for each of these in the GCP search bar and click enable to enable these API's
* Vertex AI API
***NB: all enabled in our spotted-399806 project***

### Setup GCP Credentials
Next step is to enable our container to have access to Storage buckets & Vertex AI(AI Platform) in  GCP. 

#### Create a local **secrets** folder
It is important to note that we do not want any secure information in Git. So we will manage these files outside of the git folder. At the same level as the `model-deployment` folder create a folder called **secrets**

Your folder structure should look like this:
```
   |-model-deployment
|-secrets
```

#### Setup GCP Service Account
- Here are the step to create a service account:
- To setup a service account you will need to go to [GCP Console](https://console.cloud.google.com/home/dashboard), search for  "Service accounts" from the top search box. or go to: "IAM & Admins" > "Service accounts" from the top-left menu and create a new service account called "model-deployment". For "Service account permissions" select "Storage Admin", "AI Platform Admin", "Vertex AI Administrator".
- This will create a service account
- On the right "Actions" column click the vertical ... and select "Manage keys". A prompt for Create private key for "model-deployment" will appear select "JSON" and click create. This will download a Private key json file to your computer. Copy this json file into the **secrets** folder. Rename the json file to `model-deployment.json`

***NB: there's already a model-deployment service account in spotted-399806 project, so you will only need to download json key, rename it and put it into the proper folder.***

### Create GCS Bucket

We need a bucket to store the saved model files that we will be used by Vertext AI to deploy models.
***NB: in sspotted-399806 project we have a **spotted-models-deployment** bucket that is already pre-set in docker-shell.sh***

## Run Container
### Update `docker-shell.sh` - make sure that a proper line is commented/uncommented for building image on M1/M2 vs Intel chip

### Update 'Dockerfile' - make sure that a proper line is there for tensorflow
- for Intel chip you should have under [packages] 
tensorflow = "*"
- for M1/M2 you should have 
tensorflow-aarch64 = "*"

### Pipfile.lock in file in git is the one that is created for Intel Mac machine, if you are running M1/M2 you will need to remove this one and use the one that's named Pipfile.lock.M1 (rename it to Pipfile.lock before running sh docker-shell.sh)

### Run `docker-shell.sh` or `docker-shell.bat`
Based on your OS, run the startup script to make building & running the container easy



The following params are already pre-set in the docker-shell.sh:
```
export IMAGE_NAME=model-deployment-spotted-cli
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT=spotted-399806
export GCS_MODELS_BUCKET_NAME=spotted-models-deployment
export BEST_MODEL="EfficientNetV2_finetuning_train_base_True"

# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
# M1/2 chip macs use this line
docker build -t $IMAGE_NAME --platform=linux/arm64/v8 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/model-deployment.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_MODELS_BUCKET_NAME=$GCS_MODELS_BUCKET_NAME \
$IMAGE_NAME
```

- Make sure you are inside the `model-deployment` folder and open a terminal at this location
- Run `sh docker-shell.sh` or `docker-shell.bat` for windows

### Prepare Model for Deployment
We have our model weights stored in WandB after we performed serverless training. In this step we will download the model and upload it to a GCS bucket so Vertex AI can have access to it to deploy to an endpoint.

* Run `python cli.py --upload`, this will download the model weights from WandB and upload to the specified bucket in `GCS_MODELS_BUCKET_NAME`. Please make sure that the model you want to download is defined in 'BEST_MODEL' in docker-shell.sh

### Upload & Deploy Model to Vertex AI
In this step we first upload our model to Vertex AI Model registry. Then we deploy the model as an endpoint in Vertex AI Online prediction.

* Run `python cli.py --deploy`, this option will both upload and deploy model to Vertex AI
* This will take a few minutes to complete
* Once the model has been deployed the endpoint will be displayed. The endpoint will be similar to: `projects/129349313346/locations/us-central1/endpoints/5072058134046965760`

### Test Predictions

* Update the endpoint uri in `cli.py`
* Run `python cli.py --predict`
* You  shouls see results simsilar to this:
```
Predict using endpoint
image_files: ['data/bichon_frise_1.jpg', 'data/appenzeller_1.jpg', 'data/black_sable_1.jpg', 'data/beagle_1.jpg', 'data/african_hunting_dog_1.jpg']
Prediction probabilities:  {'staffordshire_bull_terrier': 0.219, 'vizsla': 0.153, 'african_hunting_dog': 0.132}
Label:    staffordshire_bull_terrier
Image:   data/beagle_1.jpg

```