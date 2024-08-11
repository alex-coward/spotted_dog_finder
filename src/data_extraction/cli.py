"""
Command line application for performing data extraction.
"""

import os
import argparse
import indiv_data
import breed_data
from google.cloud import storage

BUCKET_ROOT = "gcs" if "GCS_BUCKET_ROOT" not in os.environ else os.environ["GCS_BUCKET_ROOT"]
RAW_BUCKET = "spot-raw-data"

BREED_DATA = {
    "stanford" : {
        "bucket_name": "stanford",
        "images_link": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
        "images_file": "images.tar",
        "annotations_link": "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar",
        "annotations_file": "annotation.tar"
    },
    "tsinghua": {
        "bucket_name": "tsinghua",
        "images_link": "https://cloud.tsinghua.edu.cn/f/80013ef29c5f42728fc8/?dl=1",
        "images_file": "low-resolution.zip",
        "annotations_link": "https://cg.cs.tsinghua.edu.cn/ThuDogs/low-annotations.zip",
        "annotations_file": "low-annotations.zip"
    }
}

# Creates a folder inside root bucket
def create_bucket_folder(name):
    folder = os.path.join("/", BUCKET_ROOT, RAW_BUCKET, name)

    if os.path.exists(folder):
        print(f'{name} folder already exists. Delete folder first to re-download dataset.')
        return False
    
    os.makedirs(folder)
    return True


# Download images and annotations for a breed dataset
def download_breed_dataset(dict):
    
    # If dataset already exists, return
    if not create_bucket_folder(dict['bucket_name']):
        return

    print(f'Extracting dataset for {dict["bucket_name"]}:')

    print("- Downloading image data from source...")
    breed_data.unarchive_file(dict['bucket_name'], dict['images_link'], dict['images_file'])

    print("- Downloading annotation data from source...")
    breed_data.unarchive_file(dict['bucket_name'], dict['annotations_link'], dict['annotations_file'])

    print('Dataset extraction complete!\n')


# Download the Austin Pets dataset and upload to GCS
def download_austin_pets():
    if not create_bucket_folder("austin"):
        return

    print('Extracting Austin Pets dataset:')
    indiv_data.download_austin_pets()
    print('Austin Pets Alive dataset extraction complete!\n')


def main(args=None):
    # Verify deestination bucket exists
    client = storage.Client()
    bucket = client.bucket(RAW_BUCKET)

    if not bucket.exists():
        print(f'ERROR: Required bucket {RAW_BUCKET} does not exist. Create first in GCS, then run again.')
        raise Exception(f'ERROR: Required bucket {RAW_BUCKET} does not exist. Create first in GCS, then run again.')

    if args.breed or args.all:
        download_breed_dataset(BREED_DATA["stanford"])
        download_breed_dataset(BREED_DATA["tsinghua"])

    if args.individual or args.all:
        download_austin_pets()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    parser = argparse.ArgumentParser(description="Data Collector CLI")

    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Download all datasets and annotations",
    )

    parser.add_argument(
        "-b",
        "--breed",
        action="store_true",
        help="Download all dog breed datasets",
    )

    parser.add_argument(
        "-i",
        "--individual",
        action="store_true",
        help="Download all individual dog datasets",
    )

    args = parser.parse_args()

    main(args)