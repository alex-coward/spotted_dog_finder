import argparse
import os
import requests
import zipfile
import tarfile
import time

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Tensorflow Hub
import tensorflow_hub as hub

import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split

# W&B
import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger

from functools import partial

# Setup the arguments for the trainer task
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-dir", dest="model_dir", default="test", type=str, help="Model dir."
)
parser.add_argument("--lr", dest="lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument(
    "--model_name",
    dest="model_name",
    default="mobilenetv2",
    type=str,
    help="Model name",
)
parser.add_argument(
    "--train_base",
    dest="train_base",
    default=False,
    action="store_true",
    help="Train base or not",
)
parser.add_argument(
    "--epochs", dest="epochs", default=10, type=int, help="Number of epochs."
)
parser.add_argument(
    "--batch_size", dest="batch_size", default=16, type=int, help="Size of a batch."
)
parser.add_argument(
    "--wandb_key", dest="wandb_key", default="16", type=str, help="WandB API Key"
)
args = parser.parse_args()

# TF Version
print("tensorflow version", tf.__version__)
print("Eager Execution Enabled:", tf.executing_eagerly())
# Get the number of replicas
strategy = tf.distribute.MirroredStrategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

devices = tf.config.experimental.get_visible_devices()
print("Devices:", devices)
print(tf.config.experimental.list_logical_devices("GPU"))

print("GPU Available: ", tf.config.list_physical_devices("GPU"))
print("All Physical Devices", tf.config.list_physical_devices())

# Login into wandb
wandb.login(key=args.wandb_key)

########################################################################
# DATA SECTION: MAKE CHANGES HERE IF YOU WANT TO USE A DIFFERENT DATASET 
########################################################################

# Load CSV data from Google Bucket
csv_of_all_images = "gs://dog-breeds-224/preprocessed/csv/all-data.csv"
df = pd.read_csv(csv_of_all_images, dtype={'filename': 'object', 'label': 'object'})

# Get unique breeds from dataset
breeds = np.unique(df['label'])

# Number of unique labels
num_classes = len(breeds)

# Create label index for easy lookup
label2index = dict((name, index) for index, name in enumerate(breeds))
index2label = dict((index, name) for index, name in enumerate(breeds))

# Helper functions for reading dataset from TFRecords
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
IMAGE_SIZE = [224, 224]

# Decode and normalize image
def decode_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.cast(image, tf.float32)
  image = tf.reshape(image, [*IMAGE_SIZE, 3])

  return image

@tf.function
def read_tfrecord(example):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
    )

    example = tf.io.parse_single_example(example, tfrecord_format)

    # Decode image from bytes
    image = decode_image(example["image"])

    # One hot encoding for lables
    label = example["label"]
    label = tf.one_hot(label, num_classes)

    return image, label

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed

    # Automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames)

    # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.with_options(ignore_order)

    # Returns a dataset of (image, label)
    dataset = dataset.map(partial(read_tfrecord), num_parallel_calls=AUTOTUNE)

    return dataset

def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(1024)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)

    return dataset

train_files = tf.io.gfile.glob("gs://dog-breeds-224/processed/train*.tfrecords")
test_files = tf.io.gfile.glob("gs://dog-breeds-224/processed/test*.tfrecords")
validation_files = tf.io.gfile.glob("gs://dog-breeds-224/processed/val*.tfrecords")

train_data = get_dataset(train_files)
validation_data = get_dataset(validation_files)
test_data = get_dataset(test_files)

# data augmentation layer to be called in the model compilation
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(224, 224,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.3),
  ]
)

#######################################################
# MODEL FUNCTION SECTION: ADD ANY MODELS YOU WANT TO 
# INCLUDE BY  ADDING A FUNCTION FOR BUILDING THE MODEL
# AS PER THE ALREADY EXISTING MODEL FUNCTIONS
#######################################################

def build_efficientnetv2_model(
    image_height, image_width, num_channels, num_classes, model_name, train_base=False
):
    # Model input
    input_shape = [image_height, image_width, num_channels]  # height, width, channels

    # Load a pretrained model from keras.applications
    transfer_model_base = keras.applications.EfficientNetV2B0(
                         include_top=False,
                         weights="imagenet",
                         input_shape = input_shape,
                         include_preprocessing=False)
                         
    # Freeze the mobileNet model layers
    transfer_model_base.trainable = train_base

    # Input Layer
    inputs = keras.Input(shape=(image_height, image_width, num_channels))

    # Scales input from (0, 255) to a range of (-1., +1.) for EfficientNet
    x = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)(inputs)

    # EfficientNet Base model (freezing batch normalization layers)
    x = transfer_model_base(x, training=False)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)

    # One nodes for categorical classification
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    full_name = model_name + "_train_base_" + str(train_base)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name=full_name)

    return model

###############################################################################
# InceptionResNetV2 model
###############################################################################
def build_InceptionResNetV2_model(
    image_height, image_width, num_channels, num_classes, model_name, train_base=False
):
    # Model input
    input_shape = [image_height, image_width, num_channels]  # height, width, channels

    # Load a pretrained model from keras.applications
    transfer_model_base = keras.applications.InceptionResNetV2(
                         include_top=False,
                         weights="imagenet",
                         input_shape = input_shape,
                         )
                         
    # Freeze the mobileNet model layers
    transfer_model_base.trainable = train_base

    # Input Layer
    inputs = keras.Input(shape=(image_height, image_width, num_channels))
    # Apply random data augmentation
    x = data_augmentation(inputs)  

    # Scales input from (0, 255) to a range of (-1., +1.) for InceptionResNetV2
    x = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)(inputs)

    # Create Base model (freezing batch normalization layers)
    x = transfer_model_base(x, training=False)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)

    # One nodes for categorical classification
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    full_name = model_name + "_train_base_" + str(train_base)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name=full_name)

    return model

###############################################################################
# THIS MODEL IS DESIGNED TO RUN FINETUNING ON A MODEL THAT IS LOADED FROM WandB.
# CHANGE w_b_run_info TO A DIFFERENT RUN TO CONTINUE TRAINING FROM THE
#  model-best.h5 FILE FOR THE RELEVANT RUN. 
###############################################################################

def build_efficientnetv2_finetuning_model():

    # Download best model from W&B run
    w_b_run_info= "spotted-dog/dog-breed-serverless-training/huzisvuo"

    api = wandb.Api()
    run = api.run(f"{w_b_run_info}")
    run.file("model-best.h5").download()

    # Load W&B model into Keras
    model = tf.keras.models.load_model('model-best.h5')
                         
    # Freeze the mobileNet model layers
    model.trainable = True
    return model


print("Train model")

#############################
def build_InceptionResNetV2_finetuning_model():

    # Download best model from W&B run
    w_b_run_info= "spotted-dog/dog-breed-serverless-training/vl3484wb"

    api = wandb.Api()
    run = api.run(f"{w_b_run_info}")
    run.file("model-best.h5").download()

    # Load W&B model into Keras
    model = tf.keras.models.load_model('model-best.h5')
                         
    # Freeze the mobileNet model layers
    model.trainable = True
    return model



###############################################################################
# Function to build student model
###############################################################################
def build_student_model(image_height, image_width, num_channels, num_classes, model_name='student'):
  # Model input
  input_shape = [image_height, image_width, num_channels]  # height, width, channels

  model = Sequential(
      [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="same",
                            kernel_initializer=keras.initializers.GlorotUniform(seed=1212)),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same",
                            kernel_initializer=keras.initializers.GlorotUniform(seed=2121)),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        keras.layers.Flatten(),
        # keras.layers.Dense(units=32, kernel_initializer=keras.initializers.GlorotUniform(seed=2323)),
        keras.layers.Dense(units=num_classes, kernel_initializer=keras.initializers.GlorotUniform(seed=3434))
      ],
      name=model_name)

  return model



############################################
# Training Params
# CHANGE AS DESIRED PRIOR TO RUNNING cli.sh
############################################

model_name = args.model_name
learning_rate = 0.001
image_width = 224
image_height = 224
num_channels = 3
batch_size = args.batch_size
epochs = args.epochs
train_base = args.train_base

# Free up memory
K.clear_session()

##########################################################
# MODEL BUILDING: CHANGE TO INCLUDE ANY DESIRED NEW MODELS
# DEFINED ABOVE IN MODEL FUNCTION SECTION
# FINETUNING MODEL CURRENT SET TO USE A DIFFERENT 
# LEARNING RATE THAT CAN BE SET BELOW
##########################################################

if model_name == "EfficientNetV2":
    # Model
    model = build_efficientnetv2_model(
        image_height,
        image_width,
        num_channels,
        num_classes,
        model_name,
        train_base=train_base,
    )
    # Optimizer
    optimizer = optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    # Loss
    loss = keras.losses.categorical_crossentropy
    # Print the model architecture
    print(model.summary())
    # Compile
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

elif model_name == "EfficientNetV2_finetuning":
    # Model
    model = build_efficientnetv2_finetuning_model()
    # Optimizer
    optimizer = keras.optimizers.Adam(1e-5)
    # Loss
    loss = keras.losses.categorical_crossentropy
    # Print the model architecture
    print(model.summary())
    # Compile
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

elif model_name == "InceptionResNetV2":
    
    # Model
    model = build_InceptionResNetV2_model(
        image_height,
        image_width,
        num_channels,
        num_classes,
        model_name,
        train_base=train_base,
    )
    # Optimizer
    optimizer = optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    # Loss
    loss = keras.losses.categorical_crossentropy
    # Print the model architecture
    print(model.summary())
    # Compile
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

elif model_name == "InceptionResNetV2_finetuning":
    # Model
    model = build_InceptionResNetV2_finetuning_model()
    # Optimizer
    optimizer = keras.optimizers.Adam(1e-5)
    # Loss
    loss = keras.losses.categorical_crossentropy
    # Print the model architecture
    print(model.summary())
    # Compile
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

elif model_name == "Student_model_scratch":
    # Model
    model = build_student_model(
        image_height,
        image_width,
        num_channels,
        num_classes,
        model_name,
        train_base=train_base,
    )
    # Optimizer
    optimizer = optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    # Loss
    loss = keras.losses.categorical_crossentropy
    # Print the model architecture
    print(model.summary())
    # Compile
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"]) 

#########
# Branch for distillation
#########
elif model_name in ("Distilled_InceptionResNetV2_finetune", "Distilled_EfficientNetV2_finetune"):
    # Build distiller
    class Distiller(Model):
        def __init__(self, teacher, student):
            super(Distiller, self).__init__()
            self.teacher = teacher
            self.student = student

        def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, Lambda = 0.1, temperature=3):
            """
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft student predictions and soft teacher predictions
            lambda: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions. Larger temperature gives softer distributions.
            """
            super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
            self.student_loss_fn = student_loss_fn
            self.distillation_loss_fn = distillation_loss_fn

            #hyper-parameters
            self.Lambda = Lambda
            self.temperature = temperature

        def train_step(self, data):
            # Unpack data
            x, y = data

            # Forward pass of teacher (professor)
            teacher_predictions = self.teacher(x, training=False)

            with tf.GradientTape() as tape:
                # Forward pass of student
                student_predictions = self.student(x, training=True)

                # Compute losses
                student_loss = self.student_loss_fn(y, student_predictions)
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                loss = self.Lambda * student_loss + (1 - self.Lambda) * distillation_loss

            # Compute gradients
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update the metrics configured in `compile()`.
            self.compiled_metrics.update_state(y, student_predictions)

            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update(
                {"student_loss": student_loss, "distillation_loss": distillation_loss}
            )
            return results

        def test_step(self, data):
            # Unpack the data
            x, y = data

            # Compute predictions
            y_prediction = self.student(x, training=False)

            # Calculate the loss
            student_loss = self.student_loss_fn(y, y_prediction)

            # Update the metrics.
            self.compiled_metrics.update_state(y, y_prediction)

            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update({"student_loss": student_loss})
            return results
    
    # download teacher model
    # Download best model from W&B run (this is finetuned EfficientNet)
    w_b_run_info= "spotted-dog/dog-breed-serverless-training/apvnifpx" 
    api = wandb.Api()
    run = api.run(f"{w_b_run_info}")
    run.file("model-best.h5").download(replace=True)
    # Load W&B model into Keras
    teacher_model = tf.keras.models.load_model('model-best.h5')
    teacher_model.summary()

    # Build student and Distil teacher to student
    ############################
    # Training Params
    ############################
    learning_rate = 0.001
    epochs = 50
    Lambda = 0.75
    temperature= 12
    loss=CategoricalCrossentropy()
    optimizer=keras.optimizers.Adam()

    # Free up memory
    K.clear_session()

    # Build Student model
    student_model = build_student_model(image_height, image_width, num_channels, num_classes, model_name='student_distill')
    print(student_model.summary())

    # Build the distiller model
    distiller_model = Distiller(teacher=teacher_model, student=student_model)

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    # Loss
    student_loss = loss
    distillation_loss = loss

    # Compile
    distiller_model.compile(
        optimizer=optimizer,
        student_loss_fn=student_loss,
        distillation_loss_fn=distillation_loss,
        metrics=["accuracy"],
        Lambda=Lambda,
        temperature=temperature
    )
    model = distiller_model

###########################################################
# BRANCHES BASED ON WHETHER YOU ARE DOING A FINETUNING
# RUN USING WandB model-best.h5 FILE AS DETERMINED 
# BY model_name ARGUMENT SET IN cli.sh or cli-multi-gpu.sh
# UPDATE PROJECT AND OTHER INFO AS PER YOUR USE CASE
############################################################

# Initialize a W&B run
if model_name == "EfficientNetV2_finetuning":
    wandb.init(
          project="dog-breed-serverless-training",
          config={
                  "learning_rate": learning_rate,
                  "epochs": epochs,
                  "batch_size": batch_size,
                  "model_name": model_name},
          name=model_name)
else:
    wandb.init(
          project="dog-breed-serverless-training",
          config={
                  "learning_rate": learning_rate,
                  "epochs": epochs,
                  "batch_size": batch_size,
                  "model_name": model.name},
          name=model.name)

# Train model
start_time = time.time()
training_results = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    callbacks=[WandbCallback()],
    verbose=1,
)
execution_time = (time.time() - start_time) / 60.0
print("Training execution time (mins)", execution_time)

# Update W&B
wandb.config.update({"execution_time": execution_time})
# Close the W&B run
wandb.run.finish()


print("Training Job Complete")