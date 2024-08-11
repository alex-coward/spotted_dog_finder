#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="spotted_data_transformation"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/

# Project Name for Google Cloud Project
export GCP_PROJECT=spotted-399806

# Select same region as Vertex AI Pipeline and GCS Buckets
export GCS_REGION="us-central1"

# Local folder to mount using GCSFuse
export GCS_BUCKET_ROOT="buckets"

# Build Docker Image
# docker buildx build --platform linux/amd64 -t $IMAGE_NAME -f Dockerfile .
docker build -t $IMAGE_NAME -f Dockerfile .

# Run Container
docker run --rm --privileged --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/data-service-account.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCS_BUCKET_ROOT=$GCS_BUCKET_ROOT \
-e GCS_REGION=$GCS_REGION \
$IMAGE_NAME