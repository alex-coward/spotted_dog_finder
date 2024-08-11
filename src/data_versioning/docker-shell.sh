#!/bin/bash

set -e

export IMAGE_NAME=dvc-docker-image
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCS_BUCKET_URI="gs://small-dogs-test-bucket"
export GCP_PROJECT="spotted"
export GCP_REGION="us-central1"

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .