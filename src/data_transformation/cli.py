import argparse
import json
import os
import shutil
import glob
import xml.etree.ElementTree as ET
from google.cloud import storage
from multiprocessing import Pool
from PIL import Image

BUCKET_ROOT = "gcs" if "GCS_BUCKET_ROOT" not in os.environ else os.environ["GCS_BUCKET_ROOT"]

POOLED_BUCKET_NAME = "spot-breed-source"
STANFORD_BUCKET_NAME = "spot-raw-data/stanford"
TSINGHUA_BUCKET_NAME = "spot-raw-data/tsinghua"


# Create folders for output
def create_folders():
    os.makedirs(f'/{BUCKET_ROOT}/{POOLED_BUCKET_NAME}/images', exist_ok=True)
    os.makedirs(f'/{BUCKET_ROOT}/{POOLED_BUCKET_NAME}/annotations', exist_ok=True)
    os.makedirs(f'/{BUCKET_ROOT}/{POOLED_BUCKET_NAME}/cropped', exist_ok=True)

    return


# Pads image to a square, resizes image (if necessary), and saves to destination (paths = [source, dest])
def resize_image_and_save(paths):

    # Verify destination image does not already exist
    if os.path.exists(paths[1]):
        return

    # Pad image so it is square
    with Image.open(paths[0]) as img:
        org_size = img.size
        new_dim = max(org_size)

        # Create a new image and paste the image into it
        new_img = Image.new("RGB", (new_dim, new_dim))
        new_img.paste(img, ((new_dim-org_size[0])//2, (new_dim-org_size[1])//2))

        # Save image to destination
        new_img.save(paths[1])

    return


# Use multiprocessing to transform sources images and save to destinations
def process_images(sources, destinations):
    data = [[s, d] for s, d in zip(sources, destinations)]

    with Pool() as pool:
        pool.map(resize_image_and_save, data)

    return


# Transform annotation base on padding and save as JSON to destination
def transform_annotation(data):
    dataset = data[1].split('-')[-2]

    # Load source XML file
    tree = ET.parse(data[0])
    root = tree.getroot()

    # Extract values from XML tree
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Get bounding box based on dataset
    if dataset == 'sd':
        body_bb = root.find('object').find('bndbox')

    elif dataset == 'ts':
        body_bb = root.find('object').find('bodybndbox')

    xmin = int(body_bb.find('xmin').text)
    ymin = int(body_bb.find('ymin').text)
    xmax = int(body_bb.find('xmax').text)
    ymax = int(body_bb.find('ymax').text)

    # Adjust bounding box for new padding and make values relative
    new_dim = max(width, height)

    v_padding = (new_dim - height) // 2
    h_padding = (new_dim - width) // 2

    xmin = float(xmin + h_padding) / new_dim
    xmax = float(xmax + h_padding) / new_dim
    ymin = float(ymin + v_padding) / new_dim
    ymax = float(ymax + v_padding) / new_dim

    # Write values to new JSON file at destination
    dictionary = {
        "width": new_dim,
        "height": new_dim,
        "xmin": round(xmin, 5),
        "ymin": round(ymin, 5),
        "xmax": round(xmax, 5),
        "ymax": round(ymax, 5)
    }

    json_object = json.dumps(dictionary, indent=4)

    with open(data[1], "w") as file:
        file.write(json_object)

    return


# Crop an image based on its bounding box from its annotation
def crop_single_image(paths):
    # Verify destination image does not already exist
    if os.path.exists(paths[1]):
        return

    # Find annotation file
    filename = paths[0].split('/')[-1]
    filename = filename.split('.')[0]
    filepath = f'/{BUCKET_ROOT}/{POOLED_BUCKET_NAME}/annotations/{filename}.json'

    # Get bounding box
    with open(filepath) as f:
        bb = json.load(f)

    with Image.open(paths[0]) as img:
        width, height = img.size
        
        xmin = int(bb['xmin'] * width)
        xmax = int(bb['xmax'] * width)
        ymin = int(bb['ymin'] * height)
        ymax = int(bb['ymax'] * height)

        crop = img.crop((xmin, ymin, xmax, ymax))

        # Pad cropped image so it is square
        crop_size = crop.size
        new_dim = max(crop_size)

        # Create a new image and paste the image into it
        new_img = Image.new("RGB", (new_dim, new_dim))
        new_img.paste(crop, ((new_dim-crop_size[0])//2, (new_dim-crop_size[1])//2))

        # Save image to destination
        new_img.save(paths[1])

    return

# Create list of all images in pooled bucket to be cropped
def crop_images():
    # Get all image files
    sources = glob.glob(f'/{BUCKET_ROOT}/{POOLED_BUCKET_NAME}/images/*.jpg')

    filenames = [f.split('/')[-1] for f in sources]
    dests = [f'/{BUCKET_ROOT}/{POOLED_BUCKET_NAME}/cropped/{f}' for f in filenames]

    data = [[s, d] for s, d in zip(sources, dests)]

    with Pool() as pool:
        pool.map(crop_single_image, data)

    return


# Flattens directory structure and renames files in Stanford Dogs dataset
def transform_stanford_dogs_images():
    img_source = f'/{BUCKET_ROOT}/{STANFORD_BUCKET_NAME}/Images'
    img_dest = f'/{BUCKET_ROOT}/{POOLED_BUCKET_NAME}/images'

    img_sources = []
    img_dests = []

    # From source, find all images and renamed destinations
    for root, _, files in os.walk(img_source):
        for filename in files:
            file_ext = filename.split(".")[-1]

            # Make sure file is a valid image
            if file_ext == 'jpg' or file_ext == 'jpeg':
                # Extract breed name from directory
                dirname = root.split('/')[-1]
                breed = dirname.split('-')[-1].lower()

                # Extract image number and remove extension from filename
                number = filename.split('_')[-1]
                number = number.split('.')[0]
                
                # Set destinaiton as renamed file
                img_sources.append(os.path.join(root, filename))
                img_dests.append(os.path.join(img_dest, f'{breed}-sd-{number}.jpg'))

    # Transform images by padding and resizing
    process_images(img_sources, img_dests)

    return


def transform_standford_dogs_annotations():
    ann_source = f'/{BUCKET_ROOT}/{STANFORD_BUCKET_NAME}/Annotation'
    ann_dest = f'/{BUCKET_ROOT}/{POOLED_BUCKET_NAME}/annotations'

    ann_sources = []
    ann_dests = []

    # From source, find all annotation files and generate renamed destinations
    for root, _, files in os.walk(ann_source):
        for filename in files:
            # Extract breed name from directory
            dirname = root.split('/')[-1]
            breed = dirname.split('-')[-1].lower()

            # Extract image number and extension from filename
            number = filename.split('_')[-1]

            # Set destinaiton as renamed file
            ann_sources.append(os.path.join(root, filename))
            ann_dests.append(os.path.join(ann_dest, f'{breed}-sd-{number}.json'))

    # Transform annotations for Stanford dataset
    data = [[s, d] for s, d in zip(ann_sources, ann_dests)]

    with Pool() as pool:
        pool.map(transform_annotation, data)

    return


# Flattens directory structure and renames files in Tsinghua Dogs dataset
def transform_tsinghua_dogs_images():
    img_source = f'/{BUCKET_ROOT}/{TSINGHUA_BUCKET_NAME}/low-resolution'
    img_dest = f'/{BUCKET_ROOT}/{POOLED_BUCKET_NAME}/images'

    img_sources = []
    img_dests = []

    # From source, find all images and renamed destinaitons
    for root, _, files in os.walk(img_source):
        for filename in files:
            file_ext = filename.split(".")[-1]

            # Make sure file is a valid image
            if file_ext == 'jpg' or file_ext == 'jpeg':
                # Extract breed name from directory
                dirname = root.split('/')[-1]
                breed = dirname.split('-')[-1].lower()
                
                # Extract image number and remove extension from filename
                number = filename.split('.')[0]

                # Set destinaiton as renamed file
                img_sources.append(os.path.join(root, filename))
                img_dests.append(os.path.join(img_dest, f'{breed}-ts-{number}.jpg'))

    # Transform images by padding and resizing
    process_images(img_sources, img_dests)

    return


def transform_tsinghua_dogs_annotations():
    ann_source = f'/{BUCKET_ROOT}/{TSINGHUA_BUCKET_NAME}/Low-Annotations'
    ann_dest = f'/{BUCKET_ROOT}/{POOLED_BUCKET_NAME}/annotations'

    ann_sources = []
    ann_dests = []

    # From source, find all annotation files and generate renamed destinations
    for root, _, files in os.walk(ann_source):
        for filename in files:
            # Extract breed name from directory
            dirname = root.split('/')[-1]
            breed = dirname.split('-')[-1].lower()

            # Extract image number and extension from filename
            number = filename.split('.')[0]

            # Set destinaiton as renamed file
            ann_sources.append(os.path.join(root, filename))
            ann_dests.append(os.path.join(ann_dest, f'{breed}-ts-{number}.json'))

    # Transform annotations for Stanford dataset
    data = [[s, d] for s, d in zip(ann_sources, ann_dests)]

    with Pool() as pool:
        pool.map(transform_annotation, data)

    return


def transform_stanford_dogs():
    print('Transforming Stanford Dogs dataset:')

    print("- Transforming Standford Dogs images...")
    transform_stanford_dogs_images()
    print("Done!")

    print("- Transforming Standford Dogs annotations...")
    transform_standford_dogs_annotations()
    print("Done!")

    print('Transformation complete!')


def transform_tsinghua_dogs():
    print('Transforming Tsinghua Dogs dataset:')

    print("- Transforming Tsinghua Dogs images...")
    transform_tsinghua_dogs_images()
    print("Done!")

    print("- Transforming Tsinghua Dogs annotations...")
    transform_tsinghua_dogs_annotations()
    print("Done!")

    print('Transformation complete!')


# Transform original files and move to Google Cloud common bucket
def main():
    # Verify deestination bucket exists
    client = storage.Client()
    bucket = client.bucket(POOLED_BUCKET_NAME)

    if not bucket.exists():
        print(f'ERROR: Required bucket {POOLED_BUCKET_NAME} does not exist. Create first in GCS, then run again.')
        raise Exception(f'ERROR: Required bucket {POOLED_BUCKET_NAME} does not exist. Create first in GCS, then run again.')
    
    # Create subfolders for output
    create_folders()

    # Transform images and annotations for breed datasets
    transform_stanford_dogs()
    #transform_tsinghua_dogs()

    # Crop images and save to GCS bucket
    print("Cropping Images...")
    crop_images()
    print("Done!")


# Generate the inputs arguments parser
if __name__ == "__main__":
    main()
