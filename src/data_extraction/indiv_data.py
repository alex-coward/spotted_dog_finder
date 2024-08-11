from google.cloud import storage
import pandas as pd
import requests
import json
import os
from urllib.parse import urlparse
import dask.bag

BUCKET_ROOT = "gcs" if "GCS_BUCKET_ROOT" not in os.environ else os.environ["GCS_BUCKET_ROOT"]
RAW_BUCKET = "spot-raw-data"

IMAGES_DIR = f'/{BUCKET_ROOT}/{RAW_BUCKET}/austin/full_quality'
JSON_DIR = f'/{BUCKET_ROOT}/{RAW_BUCKET}/austin/annotations'

def download_austin_pets():
    ## Imports dogs.csv as Pandas DataFrame, cleans up column names
    dog_infoDF = pd.read_csv('./csv/apa_dogs.csv')
    dog_infoDF.rename(columns= {'AnimalID':'id','AnimalInternal-ID':'internal_id',
                                'AnimalName':'name', 'AnimalType':'type',
                                'AnimalSex':'sex','AnimalCurrentWeightPounds':'weight_pounds',
                                'AnimalDOB':'dob', 'AnimalBreed':'breed', 'AnimalColor':'color',
                                'AnimalPattern':'pattern'}, inplace=True)

    # Imports dogs_photos.csv as Pandas DataFrame, cleans up column names, fixes web addresses that
    # are incorrectly provided in csv file
    dog_photosDF = pd.read_csv('./csv/apa_photos.csv')
    dog_photosDF.rename(columns={'AnimalInternal-ID':'internal_id', 'PhotoUrl':'url'}, inplace=True)
    dog_photosDF.loc[dog_photosDF['url'].str.contains('live-cdn.shelterluv.com'), 'url']\
         = dog_photosDF['url'].str.replace('live-cdn.shelterluv.com', 'www.shelterluv.com')

    # Creates a new column, image_file to save image file name 
    # so all pictures start with the Animal_ID along with a number based on how many
    # pictures of the dog there are
    dog_photosDF['group_rank'] = dog_photosDF.groupby('internal_id')['url'].rank()
    dog_photosDF['group_rank'] = dog_photosDF['group_rank'].astype(int).astype(str)
    dog_photosDF['image_file'] = dog_photosDF['internal_id'].astype(str) + '_' + dog_photosDF['group_rank'] + '.png'

    # Merges both DataFrames on Internal_ID, drops Group_Rank column
    # (is only used for naming image file), and changing na values to "null" for
    #  json compatibility. 
    merged_df = dog_photosDF.merge(dog_infoDF, on="internal_id")
    merged_df.drop(columns=['group_rank'], inplace=True)
    merged_df.fillna(value='null', inplace=True)

    # Creates a new data column with a tuple that includes url, image_file and a json of relevant columns
    json_columns = ['internal_id','image_file','name', 'sex', 'weight_pounds', 'dob', 'breed', 'color', 'pattern']
    merged_df['json'] = merged_df[json_columns].apply(lambda x: x.to_json(), axis=1)
    merged_df['data'] = list(zip(merged_df['url'], merged_df['image_file'], merged_df['json']))

    # Creates folders to store label/metadata jsons, images
    if not os.path.exists(JSON_DIR):
        os.makedirs(JSON_DIR)

    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    # Uses dask to parralelize download of images, annotations to "local" drive
    # (intention is to use GCP bucket via GCS Fuse)

    def download_data(data):
        try:
            url = data[0]
            image_file = data[1]
            json = data[2]

            image = requests.get(url)
            image.raise_for_status()
            
            with open(os.path.join(IMAGES_DIR, f"{image_file}"), 'wb') as file:
                file.write(image.content)

            with open(os.path.join(JSON_DIR, f"{image_file[0:-3]}json"), 'w') as json_file:
                json_file.write(json)

        except requests.RequestException as e:
            print(f"Network error for URL {url}: {str(e)}")

        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")

    bag = dask.bag.from_sequence(merged_df['data'])
    bag = bag.map(download_data)
    bag.compute(scheduler='threads')