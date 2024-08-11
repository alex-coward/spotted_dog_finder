import os
from google.cloud import storage

from transformers import AutoImageProcessor, TFViTModel
from transformers import DetrImageProcessor, DetrForObjectDetection


bucket_name = os.environ["GCS_BUCKET_NAME"]
local_model_path = "/persistent/matching_models"

bucket_object_detection_path = "matching_models/object_detection_model"
local_object_detection_path = "/persistent/matching_models/object_detection_model"

bucket_embedding_path = "matching_models/embedding_model"
local_embedding_path = "/persistent/matching_models/embedding_model"

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

###############################################
############ DOWNLOAD & LOAD MODELS ###########
###############################################

# Setup model folders
if not os.path.exists(local_model_path):
    os.mkdir(local_model_path)

if not os.path.exists(local_object_detection_path):
    os.mkdir(local_object_detection_path)

if not os.path.exists(local_embedding_path):
    os.mkdir(local_embedding_path)

def download_blob(source_blob_name, destination_file_name):
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def download_object_detection():
    object_detection_config_bucket_file_path = os.path.join(bucket_object_detection_path, "config.json")
    object_detection_config_local_file_path = os.path.join(local_object_detection_path, "config.json")
    if not os.path.isfile(object_detection_config_local_file_path):
        download_blob(object_detection_config_bucket_file_path,
                  object_detection_config_local_file_path)
    
    object_detection_preprocessor_config_bucket_file_path = os.path.join(bucket_object_detection_path, "preprocessor_config.json")
    object_detection_preprocessor_config_local_file_path = os.path.join(local_object_detection_path, "preprocessor_config.json")
    if not os.path.isfile(object_detection_preprocessor_config_local_file_path):
        download_blob(object_detection_preprocessor_config_bucket_file_path, 
                  object_detection_preprocessor_config_local_file_path)
    

    object_detection_model_bucket_file_path = os.path.join(bucket_object_detection_path, "model.safetensors")
    object_detection_model_local_file_path = os.path.join(local_object_detection_path, "model.safetensors")
    if not os.path.isfile(object_detection_model_local_file_path):
        download_blob(object_detection_model_bucket_file_path,
                  object_detection_model_local_file_path)
    return

def download_embedding_model():
    embedding_config_bucket_file_path = os.path.join(bucket_embedding_path, "config.json")
    embedding_config_local_file_path = os.path.join(local_embedding_path, "config.json")
    if not os.path.isfile(embedding_config_local_file_path):
        download_blob(embedding_config_bucket_file_path,
                  embedding_config_local_file_path)
    
    embedding_preprocessor_config_bucket_file_path = os.path.join(bucket_embedding_path, "preprocessor_config.json")
    embedding_preprocessor_config_local_file_path = os.path.join(local_embedding_path, "preprocessor_config.json")
    if not os.path.isfile(embedding_preprocessor_config_local_file_path):
        download_blob(embedding_preprocessor_config_bucket_file_path,
                  embedding_preprocessor_config_local_file_path)

    embedding_model_bucket_file_path = os.path.join(bucket_embedding_path, "tf_model.h5")
    embedding_model_local_file_path = os.path.join(local_embedding_path, "tf_model.h5")
    if not os.path.isfile(embedding_model_local_file_path):
        download_blob(embedding_model_bucket_file_path,
                  embedding_model_local_file_path)
    return

def load_object_model():
    object_processor = DetrImageProcessor.from_pretrained(local_object_detection_path)
    object_model = DetrForObjectDetection.from_pretrained(local_object_detection_path)
    
    return object_processor, object_model

def load_embedding_model():
    embedding_processor = AutoImageProcessor.from_pretrained(local_embedding_path)
    embedding_model = TFViTModel.from_pretrained(local_embedding_path)
   
    return embedding_processor, embedding_model