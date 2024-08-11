import glob
import os
import requests
import shutil
import tarfile
import tempfile
import zipfile
from google.cloud import storage
from multiprocessing import Pool

BUCKET_ROOT = "gcs" if "GCS_BUCKET_ROOT" not in os.environ else os.environ["GCS_BUCKET_ROOT"]
RAW_BUCKET = "spot-raw-data"

# Transfer single file; paths = [source_path, dest_path]
def copy_file(paths):

    # Verify that file is valid
    if os.path.isfile(paths[0]):
        try:
            shutil.copyfile(paths[0], paths[1])
        
        # If directory does not yet exist, make directories first
        except IOError as err:
            os.makedirs(os.path.dirname(paths[1]), exist_ok=True)
            shutil.copyfile(paths[0], paths[1])

    return


# Use multithreading to quickly transfer files
def transfer_files(sources, destinations):
    data = [[s, d] for s, d in zip(sources, destinations)]

    with Pool() as pool:
        pool.map(copy_file, data)

    return


# Download dataset from link and move to mounted GCS bucket
def unarchive_file(bucket_name, link, filename):
    print("Starting to download file")

    # Create temporary directory to download and extract archive to
    with tempfile.TemporaryDirectory() as tempdir:

        # Downloads compressed file from source
        try:
            archive = requests.get(link)
            archive.raise_for_status()

        except HTTPError as e:
            print("Error: Unable to download source dataset")
            print(e)

            return

        print('Done!')
        print('- Extracting files from archive...')

        # Write downloaded file to temp directory
        filepath = os.path.join(tempdir, filename)

        with open(filepath, "wb") as file:
            file.write(archive.content)

        # Uncompress file based on compression type
        ext = filename.split('.')[-1].lower()
        unzip_path = os.path.join(tempdir, 'unzip')
        os.mkdir(unzip_path)

        if ext == "zip":
            with zipfile.ZipFile(filepath, 'r') as zip_f:
                zip_f.extractall(unzip_path)

        elif ext == "tar" or ext == "gz":
            with tarfile.open(filepath, 'r') as tar_f:
                tar_f.extractall(unzip_path)

        print('Done!')
        print('- Transferring files to GCS bucket...')

        # Generate list of all source files and desired destinations
        sources = glob.glob(f'{unzip_path}/**/*', recursive=True)
        dests = [f'/{BUCKET_ROOT}/{RAW_BUCKET}/{bucket_name}{file.removeprefix(unzip_path)}' for file in sources]

        # Move all files to GCS bucket using multiprocessing
        transfer_files(sources, dests)

    print('Done!\n')

    return