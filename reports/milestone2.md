# AC215: Milestone 2

**Project Name**
Spotted!

**Team Members**
Sunil Chomal, Alex Coward, Olga Leushina, and Jonathan Sessa

**Project Description**
Our project seeks to identify lost dogs from images uploaded by spotters, and find potential matches among dogs reported missing by owners.


## Containers ##

Our current dataset contains over 20,000 images of dogs accounting for 120 breeds from multiple sources and are stored in a private Google Cloud Bucket. 

Further information and instructions for running each container can be found in their source folder:


### Data Sources ###

**src/stanford_dogs Container**
- Downloads the Stanford Dogs dataset from the source
- Extacts images, flattens folder structure, and adds labels to files
- Adds images to Google Cloud

**src/web-scraper**
- A web scraping script for downloading images from Google Image Search
- Not currently in use, but we plan on implementing to obtain more images for testing


### Data Pre-Processing ###

**src/stanford_dogs_data_preprocessing**
- Takes images from Google Cloud and resizes them preserving aspect ratio
- Creates a train/validation/test split and moves images to separate folders

**src/austin_dogs_data_preprocessing**
- Downloads images and photo information from the Austin Pets Alive dataset from the source
- Extacts images and labels from dataset and uploads to Google Cloud


### Data Modeling ###

**src/baseline_model**
- Trains an EfficientNetV2B0 model to use as a baseline for future comparisons
- Uses Stanford Dogs data (but will expand to use additional data)
- Saves model locally (but will move to Google Cloud for future use)
- Used to predict dog breed based on photo (currently ~80% breed accuracy prediction on test data)

**src/matching**
- Implements Facebook AI Similarity Search (FAISS) model for matching images
- Images are vector encoded using VIT MAE
- Will be used in our application to try and match two photos to determine if they are of the same dog


### Application ###

**src/frontend**
- Creates a framework for a React app, for future use for our web application. currently it's possible to run docker compose to build both frontend container and API container and search for a random picture from unsplash.com service


## Notebooks ##

**notebooks/eda_and_baseline_model**
- Notebook used for some preliminary EDA of Stanford Dogs dataset
- Training and evaluating the EfficientNetV2B0 baseline model
