import os

#from google.cloud import aiplatform, storage
import base64

import torch
import io
import pandas as pd
from PIL import Image
import faiss
import math
import csv


local_model_path = "/persistent/matching_models"
object_model_path = "/persistent/matching_models/object_detection_model"
embedding_model_path = "/persistent/matching_models/embedding_model"


#################################################
########## FUNCTIONS USED FOR MATCHING ##########
#################################################

# bounding images
def bound_dog(image_data, object_processor, object_model):
    original_image = Image.open(io.BytesIO(image_data))
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    inputs = object_processor(images=original_image, return_tensors="pt")
    outputs = object_model(**inputs)

    target_sizes = torch.tensor([original_image.size[::-1]])
    results = object_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    bounding_boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if object_model.config.id2label[label.item()] == "dog":
            box = [round(i, 2) for i in box.tolist()]
            bounding_boxes.append((box[0], box[1], box[2], box[3]))
            print(f"Detected {object_model.config.id2label[label.item()]} with confidence " f"{round(score.item(), 3)} at location {box}")
    if len(bounding_boxes) > 0:
        bounded_dog_image = original_image.crop(bounding_boxes[0])
        return bounded_dog_image, original_image
    else:
        return original_image, original_image

# generating embeddings
def generate_embedding(image, embedding_processor, embedding_model):
    inputs = embedding_processor(images=image, return_tensors="tf")
    outputs = embedding_model(**inputs)
    numpy_embedding = outputs.pooler_output.numpy()
    return numpy_embedding

# FAISS index creation/loading and cosine similarity search

# Setup faiss folder, declare index paths
faiss_path = "/persistent/faiss"
if not os.path.exists(faiss_path):
    os.mkdir(faiss_path)

lost_index_path = "/persistent/faiss/lost_index"

found_index_path = "/persistent/faiss/found_index"

# create or load lost index
def load_or_create_lost_index(d=1280):
    try:
        lost_index = faiss.read_index(lost_index_path)
    except:
        lost_index = faiss.IndexFlatIP(d)
    return lost_index

# create or load found index
def load_or_create_found_index(d=1280):
    try:
        found_index = faiss.read_index(found_index_path)
    except:
        found_index = faiss.IndexFlatIP(d)
    return found_index

# cosine similarity search
def cosine_siliarity_search(dog_status, numpy_embedding, k=5):
    if dog_status == "lost":
        search_faiss_index = load_or_create_found_index(d=1280)
        print("loaded or created found")

    else:
        search_faiss_index = load_or_create_lost_index(d=1280)
        print("loaded or created lost")

    faiss.normalize_L2(numpy_embedding)
    print("normalized")
    similarity, similarity_index = search_faiss_index.search(numpy_embedding, k)
    print("searched")

    return similarity, similarity_index, numpy_embedding

# Functions to load or create DataFrames
# used to save data, lookup dogs

# Setup data folder, declare df paths
data_path = "/persistent/data"
if not os.path.exists(data_path):
    os.mkdir(data_path)

lost_df_path = "/persistent/data/lost_df.csv"

found_df_path = "/persistent/data/found_df.csv"

# Function to create lost dog dataframe
def load_or_create_lost_df():
    try:
        lost_df = pd.read_csv(lost_df_path)
    except:
        lost_df = pd.DataFrame(columns=["Name", "Email", "Phone Number",
                                        "Original Image Path", "Bounded Image Path"])
    return lost_df

# Function to create found dog dataframe
def load_or_create_found_df():
    try:
        found_df = pd.read_csv(found_df_path)
    except:
        found_df = pd.DataFrame(columns=["Name", "Email", "Phone Number",
                                        "Original Image Path", "Bounded Image Path"])
    return found_df

def match_image(dog_status, image_data, object_processor, object_model,
                embedding_processor, embedding_model, similarity_threshold = .7):

    # Create bounded image
    print("bounded_image start")
    bounded_image, original_image = bound_dog(image_data, object_processor, object_model)
    print("bounded_image end")
    # Create numpy embedding
    print("generate_embedding start")
    numpy_embedding = generate_embedding(bounded_image, embedding_processor, embedding_model)
    print("generate_embedding end")
    # Perform cosine similarity search
    print("similarity_search start")
    similarity, similarity_index, normalized_embedding = cosine_siliarity_search(dog_status, numpy_embedding, k=5)
    print("similarity_search end")
    
    print("return value processing start")

    # Gets values to return to the frontend
    if dog_status == "lost":
        search_df = load_or_create_found_df()
        image_file_search_type = "found"
        print(dog_status)

    else:
        search_df = load_or_create_lost_df()
        image_file_search_type = "lost"
        print(dog_status)

    return_values = {}
    above_threshold = similarity_index[similarity>similarity_threshold]

    if len(above_threshold) != 0:
        for i, index in enumerate(above_threshold):
            search_path = f"{original_images_path}{image_file_search_type}_{index}_original.jpg"
            print(search_path)
            print(search_df)
            with open(search_path, "rb") as saved_image:
                encoded_string = base64.b64encode(saved_image.read()).decode()

            return_values[i] = {"Name": str(search_df.loc[index, 'Name']),
                                "Email": str(search_df.loc[index, 'Email']),
                                "Phone Number": str(search_df.loc[index, 'Phone Number']),
                                "Similarity": str(round(similarity[0][i], 2)),
                                "image": encoded_string}
    else:
        return_values["Response"] = "No Matching Dogs Found"
    print("return value processing end")
    return return_values, normalized_embedding, original_image, bounded_image


#################################################
############# BACKGROUND TASKS ##################
#################################################


# Adding vector embedding to FAISS index and saving index


def add_save_load_faiss_index(dog_status, normalized_embedding):

    if dog_status == "lost":
        lost_index = load_or_create_lost_index(d=1280)
        lost_index.add(normalized_embedding)
        faiss.write_index(lost_index, lost_index_path)
        index_0_length = lost_index.ntotal - 1

    else:
        found_index = load_or_create_found_index(d=1280)
        found_index.add(normalized_embedding)
        faiss.write_index(found_index, found_index_path)
        index_0_length = found_index.ntotal - 1

    return index_0_length


# Save images

saved_images_path = "/persistent/images"
if not os.path.exists(saved_images_path):
    os.mkdir(saved_images_path)

original_images_path = "/persistent/images/original/"
if not os.path.exists(original_images_path):
    os.mkdir(original_images_path)

bounded_images_path = "/persistent/images/bounded/"
if not os.path.exists(bounded_images_path):
    os.mkdir(bounded_images_path)


def save_image( dog_status, index_0_length, original_image, bounded_image):


    original_image_file_path = f"{original_images_path}{dog_status}_{index_0_length}_original.jpg"
    original_image.save(original_image_file_path)

    bounded_image_file_path = f"{bounded_images_path}{dog_status}_{index_0_length}_bounded.jpg"
    bounded_image.save(bounded_image_file_path)

    return original_image_file_path, bounded_image_file_path

# Save Data submitted by user
# image file path for original and bounded image to a dataframe
# for either lost or found dogs, based on dog_status sent by user

def save_submission_data(dog_status, name, email, phone, original_image_file_path, bounded_image_file_path):
    update_data = {"Name": [str(name)], "Email": [str(email)], "Phone Number": [str(phone)], "Original Image Path": [str(original_image_file_path)], "Bounded Image Path": [str(bounded_image_file_path)]}
    update_df = pd.DataFrame(update_data)


    if dog_status == "lost":
        save_df_path = lost_df_path
    else:
        save_df_path = found_df_path
    
    if not os.path.exists(save_df_path):
        update_df.to_csv(save_df_path, mode='w', header=True, index=False)
    
    else:
        update_df.to_csv(save_df_path, mode='a', header=False, index=False)
    
    return

# Background Task Function

def model_background_tasks(dog_status, normalized_embedding, original_image, bounded_image, name, email, phone):
    index_0_length = add_save_load_faiss_index(dog_status, normalized_embedding)
    original_image_file_path, bounded_image_file_path = save_image(dog_status, index_0_length, original_image, bounded_image)
    save_submission_data(dog_status, name, email, phone, original_image_file_path, bounded_image_file_path)
    return