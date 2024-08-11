#!/bin/bash

args="$@"
echo $args

# If GSC Mount Directory exists, then mount directory locally
if ! [ -z ${GCS_BUCKET_ROOT+x} ]; then 
  gcsfuse --key-file=$GOOGLE_APPLICATION_CREDENTIALS /$GCS_BUCKET_ROOT
fi

if [[ -z ${args} ]]; 
then
  # Authenticate gcloud using service account
  gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS

  # Set GCP Project Details
  gcloud config set project $GCP_PROJECT
  
  #/bin/bash
  pipenv shell
else
  pipenv run python $args
fi