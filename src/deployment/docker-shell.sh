#!/bin/bash

# Define some environment variables
export IMAGE_NAME="spotted-app-deployment"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCP_PROJECT=spotted-399806
export GCP_ZONE="us-central1-c"
export GOOGLE_APPLICATION_CREDENTIALS=/secrets/deployment.json
export GCS_BUCKET_NAME=spotted-models-deployment
export GCS_SERVICE_ACCOUNT="ml-workflow@spotted-399806.iam.gserviceaccount.com"
export GCP_REGION="us-central1"
export GCS_PACKAGE_URI="gs://dog-breeds-trainer"

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .
#docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v /var/run/docker.sock:/var/run/docker.sock \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$BASE_DIR/../frontend-react":/frontend-react \
-v "$BASE_DIR/../api-service":/api-service \
-v "$BASE_DIR/../data_extraction":/data_extraction \
-v "$BASE_DIR/../data_transformation":/data_transformation \
-v "$BASE_DIR/../data_preprocessing":/data_preprocessing \
-v "$BASE_DIR/../data_processing":/data_processing \
-v "$BASE_DIR/../model_training":/model_training \
-v "$BASE_DIR/../model-deployment":/model-deployment \
-e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
-e USE_GKE_GCLOUD_AUTH_PLUGIN=True \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCS_SERVICE_ACCOUNT=$GCS_SERVICE_ACCOUNT \
-e GCP_REGION=$GCP_REGION \
-e GCS_PACKAGE_URI=$GCS_PACKAGE_URI \
$IMAGE_NAME

