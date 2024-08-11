import os
import traceback
import asyncio
from google.cloud import storage


bucket_name = os.environ["GCS_BUCKET_NAME"]
local_model_path = "/persistent/matching_models"
object_model_path = "/persistent/matching_models/object_detection_model"
embedding_model_path = "/persistent/matching_models/embedding_model"

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

# Setup model folders
if not os.path.exists(local_model_path):
    os.mkdir(local_model_path)

if not os.path.exists(object_model_path):
    os.mkdir(object_model_path)

if not os.path.exists(embedding_model_path):
    os.mkdir(embedding_model_path)


def check_new_files():

    new_files = []

    for blob in bucket.list_blobs(prefix="matching_models"):
        local_file_path = os.path.join("/persistent", blob.name)

        # Check if local file exists and compare timestamps
        if os.path.exists(local_file_path):
            local_timestamp = os.path.getmtime(local_file_path)
            cloud_timestamp = blob.updated.timestamp()  # Get the last modified time of the blob

            if cloud_timestamp > local_timestamp:
                new_files.append(blob.name)
        else:
            # File does not exist locally, consider it new
            new_files.append(blob.name)

    return new_files

def download_blob(source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

class TrackerService:
    def __init__(self):
        self.timestamp = 0

    async def track(self):
        while True:
            await asyncio.sleep(60)
            print("Tracking experiments...")

            # Check if any new files
            new_files = check_new_files()

            # If there are new files, download them
            if len(new_files) > 0:
                for blob_name in new_files:
                    try:
                        download_blob(blob_name, f"/persistent/{blob_name}")
                    except:
                        print(f"Error in download of {blob_name}")
                        traceback.print_exc()
