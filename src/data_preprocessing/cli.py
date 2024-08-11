import argparse
import glob
import json
import os
import shutil
import csv
import pandas as pd
from google.cloud import storage
from multiprocessing import Pool
from PIL import Image
from sklearn.model_selection import train_test_split

BUCKET_ROOT = "gcs" if "GCS_BUCKET_ROOT" not in os.environ else os.environ["GCS_BUCKET_ROOT"]

SOURCE_BUCKET_NAME = "spot-breed-source"
DEST_BUCKET_NAME = "spot-breed-data"

# Image resizing (updated if argument present)
IMG_SIZE = 224


# Create folders for output
def create_folders():
    os.makedirs(f'/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}', exist_ok=True)
    os.makedirs(f'/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}/preprocessed', exist_ok=True)

    return


# Create train/val/test splits from data and create CSV files
def create_splits():
    csv_path = f'/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}/csv'

    # If split already exist, return file names 
    if os.path.exists(f'{csv_path}/all-data.csv'):
        print("Using existing training splits (delete CSV folder to re-create splits)")

        train_csv = pd.read_csv(f'{csv_path}/train.csv')
        train = train_csv['filename'].tolist()

        val_csv = pd.read_csv(f'{csv_path}/val.csv')
        val = val_csv['filename'].tolist()

        test_csv = pd.read_csv(f'{csv_path}/test.csv')
        test = test_csv['filename'].tolist()

        return train, val, test

    print("Creating training splits and CSV files...")
    
    # Extract filenames and labels from source images
    filenames = [f for f in os.listdir(f'/{BUCKET_ROOT}/{SOURCE_BUCKET_NAME}/images')]
    breeds = [b.split('-')[0] for b in filenames]

    # Create train/val/test splits by stratifying on dog breeds
    x_train, x_vt, y_train, y_vt = train_test_split(filenames, breeds, test_size=0.3, random_state=215, stratify=breeds)
    x_val, x_test, y_val, y_test = train_test_split(x_vt, y_vt, test_size=0.5, random_state=215, stratify=y_vt)

    split_names = ['train', 'val', 'test']
    splits = zip(split_names, [zip(x_train, y_train), zip(x_val, y_val), zip(x_test, y_test)])

    # Create CSV files for each split with filename and label
    for split, data in splits: 
        csv_file = f'/{csv_path}/{split}.csv'

        os.makedirs(csv_path, exist_ok=True)

        # Remove the CSV file if one already exists
        if os.path.exists(csv_file):
            os.remove(csv_file)

        # Write to CSV file
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            field = ["filename", "label"]
            writer.writerow(field)

            for filename, label in data:
                writer.writerow([filename, label])

    # Find all split CSV files
    os.chdir(csv_path)
    files = [i for i in glob.glob('*.{}'.format('csv'))]

    # Combine all files and export
    combined_csv = pd.concat([pd.read_csv(f) for f in files])
    combined_csv.to_csv("all-data.csv", index=False)

    print("Done!")

    return x_train, x_val, x_test


# Resizes a single image and saves it to given filepath
def resize_single_image(filepath):
    if not os.path.exists(filepath):
        filename = filepath.split('/')[-1]

        # Get 'images' or 'cropped' directory
        source = filepath.split('/')[-3]

        with Image.open(f'/{BUCKET_ROOT}/{SOURCE_BUCKET_NAME}/{source}/{filename}') as img:
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            img.save(filepath)

    return


# Resize source images to desired size and move to corresponding directory
def resize_images(train, val, test, source):
    splits = ['train', 'val', 'test']

    # Create directory for each split in destination bucket
    for d in splits:
        os.makedirs(f'/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}/preprocessed/{source}/{d}', exist_ok=True)

    for split, filenames in zip(splits, [train, val, test]):
        print(f'- Resizing {source}/{split} images...')

        # Get destination filepaths for all files in this split
        filepaths = [f'/{BUCKET_ROOT}/{DEST_BUCKET_NAME}/{IMG_SIZE}/preprocessed/{source}/{split}/{f}' for f in filenames]

        # Resize images using multiprocessing
        with Pool() as pool:
            pool.map(resize_single_image, filepaths)

        print("Done!")

    return


def main(args=None):
    # If image size was given in CLI, update variable
    if args.image_size:
        global IMG_SIZE
        IMG_SIZE = int(args.image_size)

    # Create new bucket named for desired filesize
    create_folders()
    
    # Create training splits; returns list of image filenames
    train, val, test = create_splits()

    if args.cropped:
        source = "cropped"
        resize_images(train, val, test, source)

    if args.uncropped:
        source = "images"
        resize_images(train, val, test, source)


# Generate the inputs arguments parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images and create train/validation/test splits")

    parser.add_argument(
        "-c",
        "--cropped",
        action="store_true",
        help="Download all individual dog datasets",
    )

    parser.add_argument(
        "-u",
        "--uncropped",
        action="store_true",
        help="Download all individual dog datasets",
    )

    # Optional image size (otherwise uses default)
    parser.add_argument(
        "-i",
        "--image_size",
        nargs='?',
        type=int,
        help="Optional output image size (in pixels)",
    )

    args = parser.parse_args()

    main(args)
