"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --upload
        python cli.py --deploy
        python cli.py --predict
"""

import os
import requests
import zipfile
import tarfile
import argparse
from glob import glob
import numpy as np
import base64
from google.cloud import storage
from google.cloud import aiplatform
import tensorflow as tf

# # W&B - if model is downloaded from wandb
import wandb

GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_MODELS_BUCKET_NAME = os.environ["GCS_MODELS_BUCKET_NAME"]
BEST_MODEL = os.environ["BEST_MODEL"]
ARTIFACT_URI = f"gs://{GCS_MODELS_BUCKET_NAME}/{BEST_MODEL}"

data_details = {
    "image_width": 224,
    "image_height": 224,
    "num_channels": 3,
    "num_classes": 3,
    "labels": ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_bulldog', 'american_pit_bull_terrier',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_shepherd', 'australian_terrier', 'basenji', 'basset',
       'basset_hound', 'beagle', 'bedlington_terrier',
       'bernese_mountain_dog', 'bichon_frise', 'black_and_tan_coonhound',
       'black_sable', 'blenheim_spaniel', 'bloodhound', 'bluetick',
       'border_collie', 'border_terrier', 'borzoi', 'boston_bull',
       'bouvier_des_flandres', 'boxer', 'brabancon_griffo',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cane_carso', 'cardigan', 'chesapeake_bay_retriever',
       'chihuahua', 'chinese_crested_dog', 'chinese_rural_dog', 'chow',
       'clumber', 'coated_retriever', 'coated_wheaten_terrier',
       'cocker_spaniel', 'collie', 'curly_coated_retriever',
       'dandie_dinmont', 'dhole', 'dingo', 'doberman',
       'english_cocker_spaniel', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog', 'fila braziliero',
       'flat_coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short_haired_pointer', 'german_shorthaired',
       'giant_schnauzer', 'golden_retriever', 'gordon_setter',
       'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog',
       'groenendael', 'haired_fox_terrier', 'haired_pointer', 'havanese',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_chin', 'japanese_spaniel', 'japanese_spitzes',
       'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz',
       'labrador_retriever', 'lakeland_terrier', 'leonberg', 'leonberger',
       'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless',
       'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer',
       'newfoundland', 'norfolk_terrier', 'norwegian_elkhound',
       'norwich_terrier', 'old_english_sheepdog', 'otterhound',
       'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone',
       'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki',
       'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound',
       'scottish_terrier', 'sealyham_terrier', 'shetland_sheepdog',
       'shiba_dog', 'shiba_inu', 'shih_tzu', 'siberian_husky',
       'silky_terrier', 'soft_coated_wheaten_terrier',
       'staffordshire_bull_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tan_coonhound', 'teddy', 'tibetan_mastiff', 'tibetan_terrier',
       'toy_poodle', 'toy_terrier', 'tzu', 'vizsla', 'walker_hound',
       'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'wheaten_terrier', 'whippet',
       'wire_haired_fox_terrier', 'yorkshire_terrier'],
    "label2index": {'affenpinscher': 0,
        'afghan_hound': 1,
        'african_hunting_dog': 2,
        'airedale': 3,
        'american_bulldog': 4,
        'american_pit_bull_terrier': 5,
        'american_staffordshire_terrier': 6,
        'appenzeller': 7,
        'australian_shepherd': 8,
        'australian_terrier': 9,
        'basenji': 10,
        'basset': 11,
        'basset_hound': 12,
        'beagle': 13,
        'bedlington_terrier': 14,
        'bernese_mountain_dog': 15,
        'bichon_frise': 16,
        'black_and_tan_coonhound': 17,
        'black_sable': 18,
        'blenheim_spaniel': 19,
        'bloodhound': 20,
        'bluetick': 21,
        'border_collie': 22,
        'border_terrier': 23,
        'borzoi': 24,
        'boston_bull': 25,
        'bouvier_des_flandres': 26,
        'boxer': 27,
        'brabancon_griffo': 28,
        'brabancon_griffon': 29,
        'briard': 30,
        'brittany_spaniel': 31,
        'bull_mastiff': 32,
        'cairn': 33,
        'cane_carso': 34,
        'cardigan': 35,
        'chesapeake_bay_retriever': 36,
        'chihuahua': 37,
        'chinese_crested_dog': 38,
        'chinese_rural_dog': 39,
        'chow': 40,
        'clumber': 41,
        'coated_retriever': 42,
        'coated_wheaten_terrier': 43,
        'cocker_spaniel': 44,
        'collie': 45,
        'curly_coated_retriever': 46,
        'dandie_dinmont': 47,
        'dhole': 48,
        'dingo': 49,
        'doberman': 50,
        'english_cocker_spaniel': 51,
        'english_foxhound': 52,
        'english_setter': 53,
        'english_springer': 54,
        'entlebucher': 55,
        'eskimo_dog': 56,
        'fila braziliero': 57,
        'flat_coated_retriever': 58,
        'french_bulldog': 59,
        'german_shepherd': 60,
        'german_short_haired_pointer': 61,
        'german_shorthaired': 62,
        'giant_schnauzer': 63,
        'golden_retriever': 64,
        'gordon_setter': 65,
        'great_dane': 66,
        'great_pyrenees': 67,
        'greater_swiss_mountain_dog': 68,
        'groenendael': 69,
        'haired_fox_terrier': 70,
        'haired_pointer': 71,
        'havanese': 72,
        'ibizan_hound': 73,
        'irish_setter': 74,
        'irish_terrier': 75,
        'irish_water_spaniel': 76,
        'irish_wolfhound': 77,
        'italian_greyhound': 78,
        'japanese_chin': 79,
        'japanese_spaniel': 80,
        'japanese_spitzes': 81,
        'keeshond': 82,
        'kelpie': 83,
        'kerry_blue_terrier': 84,
        'komondor': 85,
        'kuvasz': 86,
        'labrador_retriever': 87,
        'lakeland_terrier': 88,
        'leonberg': 89,
        'leonberger': 90,
        'lhasa': 91,
        'malamute': 92,
        'malinois': 93,
        'maltese_dog': 94,
        'mexican_hairless': 95,
        'miniature_pinscher': 96,
        'miniature_poodle': 97,
        'miniature_schnauzer': 98,
        'newfoundland': 99,
        'norfolk_terrier': 100,
        'norwegian_elkhound': 101,
        'norwich_terrier': 102,
        'old_english_sheepdog': 103,
        'otterhound': 104,
        'papillon': 105,
        'pekinese': 106,
        'pembroke': 107,
        'pomeranian': 108,
        'pug': 109,
        'redbone': 110,
        'rhodesian_ridgeback': 111,
        'rottweiler': 112,
        'saint_bernard': 113,
        'saluki': 114,
        'samoyed': 115,
        'schipperke': 116,
        'scotch_terrier': 117,
        'scottish_deerhound': 118,
        'scottish_terrier': 119,
        'sealyham_terrier': 120,
        'shetland_sheepdog': 121,
        'shiba_dog': 122,
        'shiba_inu': 123,
        'shih_tzu': 124,
        'siberian_husky': 125,
        'silky_terrier': 126,
        'soft_coated_wheaten_terrier': 127,
        'staffordshire_bull_terrier': 128,
        'staffordshire_bullterrier': 129,
        'standard_poodle': 130,
        'standard_schnauzer': 131,
        'sussex_spaniel': 132,
        'tan_coonhound': 133,
        'teddy': 134,
        'tibetan_mastiff': 135,
        'tibetan_terrier': 136,
        'toy_poodle': 137,
        'toy_terrier': 138,
        'tzu': 139,
        'vizsla': 140,
        'walker_hound': 141,
        'weimaraner': 142,
        'welsh_springer_spaniel': 143,
        'west_highland_white_terrier': 144,
        'wheaten_terrier': 145,
        'whippet': 146,
        'wire_haired_fox_terrier': 147,
        'yorkshire_terrier': 148
    },
    "index2label": {0: 'affenpinscher',
        1: 'afghan_hound',
        2: 'african_hunting_dog',
        3: 'airedale',
        4: 'american_bulldog',
        5: 'american_pit_bull_terrier',
        6: 'american_staffordshire_terrier',
        7: 'appenzeller',
        8: 'australian_shepherd',
        9: 'australian_terrier',
        10: 'basenji',
        11: 'basset',
        12: 'basset_hound',
        13: 'beagle',
        14: 'bedlington_terrier',
        15: 'bernese_mountain_dog',
        16: 'bichon_frise',
        17: 'black_and_tan_coonhound',
        18: 'black_sable',
        19: 'blenheim_spaniel',
        20: 'bloodhound',
        21: 'bluetick',
        22: 'border_collie',
        23: 'border_terrier',
        24: 'borzoi',
        25: 'boston_bull',
        26: 'bouvier_des_flandres',
        27: 'boxer',
        28: 'brabancon_griffo',
        29: 'brabancon_griffon',
        30: 'briard',
        31: 'brittany_spaniel',
        32: 'bull_mastiff',
        33: 'cairn',
        34: 'cane_carso',
        35: 'cardigan',
        36: 'chesapeake_bay_retriever',
        37: 'chihuahua',
        38: 'chinese_crested_dog',
        39: 'chinese_rural_dog',
        40: 'chow',
        41: 'clumber',
        42: 'coated_retriever',
        43: 'coated_wheaten_terrier',
        44: 'cocker_spaniel',
        45: 'collie',
        46: 'curly_coated_retriever',
        47: 'dandie_dinmont',
        48: 'dhole',
        49: 'dingo',
        50: 'doberman',
        51: 'english_cocker_spaniel',
        52: 'english_foxhound',
        53: 'english_setter',
        54: 'english_springer',
        55: 'entlebucher',
        56: 'eskimo_dog',
        57: 'fila braziliero',
        58: 'flat_coated_retriever',
        59: 'french_bulldog',
        60: 'german_shepherd',
        61: 'german_short_haired_pointer',
        62: 'german_shorthaired',
        63: 'giant_schnauzer',
        64: 'golden_retriever',
        65: 'gordon_setter',
        66: 'great_dane',
        67: 'great_pyrenees',
        68: 'greater_swiss_mountain_dog',
        69: 'groenendael',
        70: 'haired_fox_terrier',
        71: 'haired_pointer',
        72: 'havanese',
        73: 'ibizan_hound',
        74: 'irish_setter',
        75: 'irish_terrier',
        76: 'irish_water_spaniel',
        77: 'irish_wolfhound',
        78: 'italian_greyhound',
        79: 'japanese_chin',
        80: 'japanese_spaniel',
        81: 'japanese_spitzes',
        82: 'keeshond',
        83: 'kelpie',
        84: 'kerry_blue_terrier',
        85: 'komondor',
        86: 'kuvasz',
        87: 'labrador_retriever',
        88: 'lakeland_terrier',
        89: 'leonberg',
        90: 'leonberger',
        91: 'lhasa',
        92: 'malamute',
        93: 'malinois',
        94: 'maltese_dog',
        95: 'mexican_hairless',
        96: 'miniature_pinscher',
        97: 'miniature_poodle',
        98: 'miniature_schnauzer',
        99: 'newfoundland',
        100: 'norfolk_terrier',
        101: 'norwegian_elkhound',
        102: 'norwich_terrier',
        103: 'old_english_sheepdog',
        104: 'otterhound',
        105: 'papillon',
        106: 'pekinese',
        107: 'pembroke',
        108: 'pomeranian',
        109: 'pug',
        110: 'redbone',
        111: 'rhodesian_ridgeback',
        112: 'rottweiler',
        113: 'saint_bernard',
        114: 'saluki',
        115: 'samoyed',
        116: 'schipperke',
        117: 'scotch_terrier',
        118: 'scottish_deerhound',
        119: 'scottish_terrier',
        120: 'sealyham_terrier',
        121: 'shetland_sheepdog',
        122: 'shiba_dog',
        123: 'shiba_inu',
        124: 'shih_tzu',
        125: 'siberian_husky',
        126: 'silky_terrier',
        127: 'soft_coated_wheaten_terrier',
        128: 'staffordshire_bull_terrier',
        129: 'staffordshire_bullterrier',
        130: 'standard_poodle',
        131: 'standard_schnauzer',
        132: 'sussex_spaniel',
        133: 'tan_coonhound',
        134: 'teddy',
        135: 'tibetan_mastiff',
        136: 'tibetan_terrier',
        137: 'toy_poodle',
        138: 'toy_terrier',
        139: 'tzu',
        140: 'vizsla',
        141: 'walker_hound',
        142: 'weimaraner',
        143: 'welsh_springer_spaniel',
        144: 'west_highland_white_terrier',
        145: 'wheaten_terrier',
        146: 'whippet',
        147: 'wire_haired_fox_terrier',
        148: 'yorkshire_terrier'
    },
}


def download_file(packet_url, base_path="", extract=False, headers=None):
    if base_path != "":
        if not os.path.exists(base_path):
            os.mkdir(base_path)
    packet_file = os.path.basename(packet_url)
    with requests.get(packet_url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(os.path.join(base_path, packet_file), "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    if extract:
        if packet_file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(base_path, packet_file)) as zfile:
                zfile.extractall(base_path)
        else:
            packet_name = packet_file.split(".")[0]
            with tarfile.open(os.path.join(base_path, packet_file)) as tfile:
                tfile.extractall(base_path)


def main(args=None):
    if args.upload:
        print("Upload model to GCS")

        storage_client = storage.Client(project=GCP_PROJECT)
        bucket = storage_client.get_bucket(GCS_MODELS_BUCKET_NAME)

        # Use this code if you want to pull your model directly from WandB
        WANDB_KEY = os.environ["WANDB_KEY"]
        # # Login into wandb
        wandb.login(key=WANDB_KEY)

        # # Download model artifact from wandb
        w_b_run_info= BEST_MODEL
        api = wandb.Api()
        run = api.run(f"{w_b_run_info}")

        #Setup model folder
        local_model_path = "/persistent/model"
        if not os.path.exists(local_model_path):
            os.mkdir(local_model_path)
        os.chdir(local_model_path)
        run.file("model-best.h5").download(replace = True)
        #print(os.path.join(local_model_path,"model-best.h5"))

        # Download model - github download
        #download_file(
        #    "https://github.com/dlops-io/models/releases/download/v2.0/model-mobilenetv2_train_base_True.v74.zip",
        #    base_path="artifacts",
        #    extract=True,
        #)
        #artifact_dir = "./artifacts/model-mobilenetv2_train_base_True:v74"

        # Load model
        prediction_model = tf.keras.models.load_model("model-best.h5")
        print(prediction_model.summary())

        # Preprocess Image
        def preprocess_image(bytes_input):
            decoded = tf.io.decode_jpeg(bytes_input, channels=3)
            decoded = tf.image.convert_image_dtype(decoded, tf.float32)
            resized = tf.image.resize(decoded, size=(224, 224))
            return resized

        @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
        def preprocess_function(bytes_inputs):
            decoded_images = tf.map_fn(
                preprocess_image, bytes_inputs, dtype=tf.float32, back_prop=False
            )
            return {"model_input": decoded_images}

        @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
        def serving_function(bytes_inputs):
            images = preprocess_function(bytes_inputs)
            results = model_call(**images)
            return results

        model_call = tf.function(prediction_model.call).get_concrete_function(
            [
                tf.TensorSpec(
                    shape=[None, 224, 224, 3], dtype=tf.float32, name="model_input"
                )
            ]
        )

        # Save updated model to GCS
        tf.saved_model.save(
            prediction_model,
            ARTIFACT_URI,
            signatures={"serving_default": serving_function},
        )

    elif args.deploy:
        print("Deploy model")

        # List of prebuilt containers for prediction
        # https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
        serving_container_image_uri = (
            "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
        )

        # Upload and Deploy model to Vertex AI
        # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_upload
        deployed_model = aiplatform.Model.upload(
            display_name=BEST_MODEL,
            artifact_uri=ARTIFACT_URI,
            serving_container_image_uri=serving_container_image_uri,
        )
        print("deployed_model:", deployed_model)
        # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_deploy
        endpoint = deployed_model.deploy(
            deployed_model_display_name=BEST_MODEL,
            traffic_split={"0": 100},
            machine_type="n1-standard-4",
            accelerator_count=0,
            min_replica_count=1,
            max_replica_count=1,
            sync=False,
        )
        print("endpoint:", endpoint)

    elif args.predict:
        print("Predict using endpoint")

        # Get the endpoint
        # Endpoint format: endpoint_name="projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/{ENDPOINT_ID}"
        endpoint = aiplatform.Endpoint(
            "projects/505898030155/locations/us-central1/endpoints/401860504835850240"
        )

        # Get a sample image to predict
        image_files = glob(os.path.join("data", "*.jpg"))
        print("image_files:", image_files[:5])

        image_samples = np.random.randint(0, high=len(image_files) - 1, size=1)
 
        for img_idx in image_samples:
            #print("Image:", image_files[img_idx])
            with open(image_files[img_idx], "rb") as f:
                data = f.read()
                b64str = base64.b64encode(data).decode("utf-8")
                # The format of each instance should conform to the deployed model's prediction input schema.
                instances = [{"bytes_inputs": {"b64": b64str}}]
                result = endpoint.predict(instances=instances)
                #print("Result:", result)
                prediction = result.predictions[0]
                #print(prediction, prediction.index(max(prediction)))
                max_3_labels = [data_details["index2label"][prediction.index(n)] for n in sorted(prediction, reverse=True)[:3]]
                probabilities = sorted(prediction, reverse=True)[:3]
                #print(max_3_labels)
                #print(probabilities)
                result = {max_3_labels[i]: round(probabilities[i],3) for i in range(len(max_3_labels))}
                print("Prediction probabilities: ", result)
                print(
                    "Label:   ",
                    data_details["index2label"][prediction.index(max(prediction))]
                )
                print(
                    "Image:  ",
                    image_files[img_idx],
                    "\n",
                )


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Data Collector CLI")

    parser.add_argument(
        "-u",
        "--upload",
        action="store_true",
        help="Upload saved model to GCS Bucket",
    )
    parser.add_argument(
        "-d",
        "--deploy",
        action="store_true",
        help="Deploy saved model to Vertex AI",
    )
    parser.add_argument(
        "-p",
        "--predict",
        action="store_true",
        help="Make prediction using the endpoint from Vertex AI",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Test deployment to Vertex AI",
    )

    args = parser.parse_args()

    main(args)


