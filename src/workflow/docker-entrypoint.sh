#!/bin/bash

# Authenticate gcloud using service account
gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS

# Dynamically mount GCS to local directory
#gcsfuse --key-file=$GOOGLE_APPLICATION_CREDENTIALS /$GCS_BUCKET_NAME

# Set GCP Project Details
gcloud config set project $GCP_PROJECT

# Configure GCR
gcloud auth configure-docker gcr.io -q

#/bin/bash
pipenv shell