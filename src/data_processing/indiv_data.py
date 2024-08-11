import os
import argparse
import shutil
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split

BUCKET_ROOT = "gcs" if "GCS_BUCKET_ROOT" not in os.environ else os.environ["GCS_BUCKET_ROOT"]
SOURCE_BUCKET_NAME = "spot-individual-data"

def process_indiv_data(IMG_SIZE):

    base_path = f"/{BUCKET_ROOT}/spot-raw-data/austin/full_quality"

    label_names = [label.split("_")[0] for label in os.listdir(base_path)]

    image_width = IMG_SIZE
    image_height = IMG_SIZE
    num_channels = 3

    # Create label index for easy lookup
    label_set = set(label_names)
    label2index = dict((name, index) for index, name in enumerate(label_set))
    index2label = dict((index, name) for index, name in enumerate(label_set))

    # Generate a list of labels and path to images
    data_list = []
    image_paths = [os.path.join(base_path, file) for file in os.listdir(base_path)]
    data_list.extend(zip(label_names, image_paths))

    def create_tf_example(item):
        try:
            # Read image
            image = tf.io.read_file(item[1])
            image = tf.image.decode_png(image, channels=num_channels)
            image = tf.image.resize(image, [image_height, image_width])
            image = tf.cast(image, tf.uint8)

            # Labels
            filename = item[1].split('/')[-1]
            dog_id = filename.split('_')[0]
            img_id = filename.split('_')[1]
            img_id = img_id.split('.')[0]

            # Build feature dict (used for obejct detection)
            feature_dict = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy().tobytes()])),
                'dog_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(dog_id)])),
                'img_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(img_id)])),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            return example

        except Exception as e:
            print(f"Error on image {item[1]}: {str(e)}")

    def create_tf_records(data,num_shards=10, prefix='', folder='data'):
        num_records = len(data)
        step_size = num_records//num_shards + 1

        for i in range(0, num_records, step_size):
            print("Creating shard:",(i//step_size)," from records:",i,"to",(i+step_size))
            path = '{}/{}_000{}.tfrecords'.format(folder, prefix, i//step_size)
            print(path)

            # Write the file
            with tf.io.TFRecordWriter(path) as writer:
                # Filter the subset of data to write to tfrecord file
                for item in data[i:i+step_size]:
                    try:
                        tf_example = create_tf_example(item)
                        writer.write(tf_example.SerializeToString())
                    except Exception as e:
                          print(f"Error on image {item[1]}: {str(e)}")

    # Split data into train / validate
    train_xy, validate_xy = train_test_split(data_list, test_size=.2, random_state=215)

    test_xy, validate_xy = train_test_split(validate_xy, test_size=.5, random_state=215)

    # Create an output path to store the tfrecords
    tfrecords_output_dir = f"/{BUCKET_ROOT}/{SOURCE_BUCKET_NAME}/{IMG_SIZE}/processed"
    if os.path.exists(tfrecords_output_dir):
        shutil.rmtree(tfrecords_output_dir)
    tf.io.gfile.makedirs(tfrecords_output_dir)

    # Split data into multiple TFRecord shards between 100MB to 200MB
    num_shards = 47

    # Create TF Records for train
    start_time = time.time()
    create_tf_records(train_xy,num_shards=num_shards, prefix="train", folder=tfrecords_output_dir)
    execution_time = (time.time() - start_time)/60.0
    print("Execution time (mins)",execution_time)

    # Split data into multiple TFRecord shards between 100MB to 200MB
    num_shards = 6

    # Create TF Records for test
    start_time = time.time()
    create_tf_records(validate_xy,num_shards=num_shards, prefix="test", folder=tfrecords_output_dir)
    execution_time = (time.time() - start_time)/60.0
    print("Execution time (mins)",execution_time)

    # Split data into multiple TFRecord shards between 100MB to 200MB
    num_shards = 6

    # Create TF Records for validation
    start_time = time.time()
    create_tf_records(validate_xy,num_shards=num_shards, prefix="val", folder=tfrecords_output_dir)
    execution_time = (time.time() - start_time)/60.0
    print("Execution time (mins)",execution_time)
