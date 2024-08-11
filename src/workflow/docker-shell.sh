#!/bin/bash

set -e

export IMAGE_NAME="spotted-data-pipeline-workflow"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export GCP_PROJECT=spotted-399806
export GCS_BUCKET_NAME=spotted-pipelines1
export GCS_SERVICE_ACCOUNT="ml-workflow@spotted-399806.iam.gserviceaccount.com"
export GCS_REGION="us-central1"
export GCS_PACKAGE_URI="gs://dog-breeds-trainer"
#export GCS_BUCKET_URI="gs://dog-breeds-trainer"

# Build the image based on the Dockerfile
docker buildx build --platform linux/amd64 -t $IMAGE_NAME -f Dockerfile .

# Run Container
docker run --rm --privileged --name $IMAGE_NAME -ti \
-v /var/run/docker.sock:/var/run/docker.sock \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$BASE_DIR/../data_extraction":/data_extraction \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ml-workflow.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCS_SERVICE_ACCOUNT=$GCS_SERVICE_ACCOUNT \
-e GCS_REGION=$GCS_REGION \
-e GCS_PACKAGE_URI=$GCS_PACKAGE_URI \
$IMAGE_NAME