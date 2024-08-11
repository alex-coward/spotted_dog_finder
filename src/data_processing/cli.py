import argparse
import json
import os
import shutil
import tensorflow as tf
import pandas as pd
import numpy as np
from google.cloud import storage
from multiprocessing import Pool
import indiv_data

BUCKET_ROOT = "gcs" if "GCS_BUCKET_ROOT" not in os.environ else os.environ["GCS_BUCKET_ROOT"]


POOLED_BUCKET_NAME = "spot-breed-source"
SOURCE_BUCKET_NAME = "spot-breed-data"


IMG_SIZE = 224

# Dictionary for encoding dog breed labels
label2index = {}


# Create encoding dictionary from unique breeds
def create_dictionary():
    global label2index

    # Load CSV data from Google Bucket
    csv_of_all_images = f"/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}/csv/all-data.csv"
    df = pd.read_csv(csv_of_all_images, dtype={'filename': 'object', 'label': 'object'})

    # Get unique breeds from dataset
    breeds = np.unique(df['label'])

    # Create label index for easy lookup
    label2index = dict((name, index) for index, name in enumerate(breeds))


# Create TRF Example from (image_path, label)
def get_long_tf_example_from(item):
    try:
        # Read image
        image = tf.io.decode_jpeg(tf.io.read_file(item[0]))

        breed = item[1]
        index = label2index[breed]
        
        # Find and load JSON annotation
        filename = item[0].split('/')[-1]
        filename = filename.split('.')[0]
        filepath = f'/{BUCKET_ROOT}/{POOLED_BUCKET_NAME}/annotations/{filename}.json'

        with open(filepath) as f:
            bb = json.load(f)

        # Build feature dict (used for obejct detection)
        feature_dict = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'breed': tf.train.Feature(bytes_list=tf.train.BytesList(value=[breed.encode()])),
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[IMG_SIZE])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[IMG_SIZE])),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=["dog".encode()])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=[bb['xmin']])),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=[bb['ymin']])),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=[bb['xmax']])),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=[bb['ymax']]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        
        return example

    except Exception as e:
        print(f"Error on image {item[1]}: {str(e)}")


# Create TRF Example from (image_path, label)
def get_short_tf_example_from(item):
    try:
        # Read image
        image = tf.io.decode_jpeg(tf.io.read_file(item[0]))

        breed = item[1]
        index = label2index[breed]

        # Build feature dict (used for obejct detection)
        feature_dict = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'breed': tf.train.Feature(bytes_list=tf.train.BytesList(value=[breed.encode()]))
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        
        return example

    except Exception as e:
        print(f"Error on image {item[1]}: {str(e)}")


# Give each thread: (1) list of source files and labels, (2) path to output TFRecord file
def write_tf_record(imports):
    items = imports[0]
    path = imports[1]

    source = path.split('/')[-2]

    # Write the file
    with tf.io.TFRecordWriter(path) as writer:
        for item in items:
            try:
                if source == "cropped":
                    tf_example = get_short_tf_example_from(item)
                else:
                    tf_example = get_long_tf_example_from(item)
                
                writer.write(tf_example.SerializeToString())
            
            except Exception as e:
                print(f"Error on image {item[0]}: {str(e)}")

    print(f'- Finished writing {path}')


# Create a set of TFRecord files for a set of given data
def generate_tf_records_from(data, num_shards, source, split):
    num_records = len(data)
    step_size = num_records//num_shards + 1

    shards = []

    for i in range(0, num_records, step_size):
        filename = "%s_%.3i.tfrecords" % (split, i//step_size)
        path = f'/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}/processed/{source}/{filename}'

        items = data[i:i+step_size]

        shards.append([items, path])

    # Write TFRecords files using multi-threading
    with Pool() as pool:
        pool.map(write_tf_record, shards)

    return


# Generate a complete set of TFRecords for train/val/test images in source bucket
def create_tf_records_for(source):
    # Create bucket to store TFRecords
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f'Creating TFRecords for {split}...')

        # Create data for split
        source_dir = f'/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}/preprocessed/{source}/{split}'

        filenames = [f for f in os.listdir(source_dir)]
        filepaths = [f'{source_dir}/{f}' for f in filenames]

        labels = [f.split('-')[0] for f in filenames]

        # Place 1000 images in each shard
        img_per_shard = 5000 if IMG_SIZE < 300 else 1000
        num_shards = len(filepaths) // img_per_shard
        num_shards = num_shards if num_shards>0 else 1

        data = []
        data.extend(zip(filepaths, labels))

        generate_tf_records_from(data, num_shards, source, split)

    return


# Transform original files and move to Google Cloud common bucket
def main(args=None):
    if args.image_size:
        try:
            global IMG_SIZE
            IMG_SIZE = int(args.image_size)
        except:
            print("Warning: Image size not found from bucket name, using default instead.")

    #if args.individual:
    #    indiv_data.process_indiv_data(IMG_SIZE)

    if args.uncropped:
        # Create label dictionary for dog breeds
        create_dictionary()

        source = "images"
        os.makedirs(f'/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}/processed/{source}', exist_ok=True)
        os.makedirs(f'/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}/processed/{source}/dvc_store', exist_ok=True)

        create_tf_records_for(source)

    if args.cropped:
        # Create label dictionary for dog breeds
        create_dictionary()

        source = "cropped"
        os.makedirs(f'/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}/processed/{source}', exist_ok=True)
        os.makedirs(f'/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}/processed/{source}/dvc_store', exist_ok=True)

        create_tf_records_for(source)

    else:
        print("Required bucket name argument missing. Use '-h' for help.")


# Generate the inputs arguments parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create TFRecord files ")

    # Process cropped breed images
    parser.add_argument(
        "-c",
        "--cropped",
        action="store_true",
        help="Download all individual dog datasets",
    )

    # Process uncropped breed images
    parser.add_argument(
        "-u",
        "--uncropped",
        action="store_true",
        help="Download all individual dog datasets",
    )

    # Process individual dog images
    #parser.add_argument(
    #    "-a",
    #    "--individual",
    #    action="store_true",
    #    help="Download all individual dog datasets",
    #)

    # Optional image size (otherwise uses default=224)
    parser.add_argument(
        "-i",
        "--image_size",
        nargs='?',
        type=int,
        help="Optional output image size (in pixels)",
    )

    args = parser.parse_args()

    main(args)
