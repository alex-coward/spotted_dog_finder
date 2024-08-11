"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py
"""

import os
import argparse
import random
import string
from kfp import dsl
from kfp import compiler
import google.cloud.aiplatform as aip
from google.cloud import storage
from model import model_training, model_deploy

GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_REGION = os.environ["GCS_REGION"]
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]

GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
GCS_PACKAGE_URI = os.environ["GCS_PACKAGE_URI"]
BUCKET_URI = f"gs://spotted-pipelines1"
PIPELINE_ROOT = f"{BUCKET_URI}"

# Docker Hub Images

DATA_EXTRACTION_IMAGE = "oll583921/spotted-data-extraction"
DATA_TRANSFORMATION_IMAGE = "oll583921/spotted-data-transformation"
DATA_PREPROCESSING_IMAGE = "oll583921/spotted-data-preprocessing"
DATA_PROCESSING_IMAGE = "oll583921/spotted-data-processing"
MODEL_TRAIN_IMAGE = "oll583921/spotted-model-training"
MODEL_DEPLOYMENT_IMAGE = "oll583921/spotted-model-deployment"


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

def get_arguments(args, container):
    if container == "data_extraction":
        return ["cli.py", "--all"]

    if container == "data_transformation":
        return ["cli.py"]

    if container == "data_preprocessing":
        arg_list = ["cli.py"]

        if args.image_size:
            arg_list.append(f"--image_size {args.image_size}")

        if args.cropped:
            arg_list.append("--cropped")

        if args.uncropped:
            arg_list.append("--uncropped")

        return arg_list

    if container == "data_processing":
        arg_list = ["cli.py"]

        if args.image_size:
            arg_list.append(f"--image_size {args.image_size}")

        if args.cropped:
            arg_list.append("--cropped")

        if args.uncropped:
            arg_list.append("--uncropped")

        #if args.individual:
        #    arg_list.append("--individual")

        return arg_list


def main(args=None):
    print("CLI Arguments:", args)

    if args.data_extraction:
        # Define a Container Component
        @dsl.container_component
        def data_extraction():
            container_spec = dsl.ContainerSpec(
                image=DATA_EXTRACTION_IMAGE,
                command=[],
                args=get_arguments(args, "data_extraction"),
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def data_extraction_pipeline():
            data_extraction()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_extraction_pipeline, package_path="data_extraction.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-data-extraction-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_extraction.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.submit(service_account=GCS_SERVICE_ACCOUNT)

    if args.data_transformation:
        # Define a Container Component
        @dsl.container_component
        def data_transformation():
            container_spec = dsl.ContainerSpec(
                image=DATA_TRANSFORMATION_IMAGE,
                command=[],
                args=get_arguments(args, "data_transformation"),
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def data_transformation_pipeline():
            data_transformation().set_cpu_limit("32")

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_transformation_pipeline, package_path="data_transformation.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-data-transformation-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_transformation.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.submit(service_account=GCS_SERVICE_ACCOUNT)

    if args.data_preprocessing:
        # Define a Container Component
        @dsl.container_component
        def data_preprocessing():
            container_spec = dsl.ContainerSpec(
                image=DATA_PREPROCESSING_IMAGE,
                command=[],
                args=get_arguments(args, "data_preprocessing"),
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def data_preprocessing_pipeline():
            data_preprocessing().set_cpu_limit("32")

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_preprocessing_pipeline, package_path="data_preprocessing.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-data-preprocessing-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_preprocessing.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.submit(service_account=GCS_SERVICE_ACCOUNT)

    if args.data_processing:
        # Define a Container Component
        @dsl.container_component
        def data_processing():
            container_spec = dsl.ContainerSpec(
                image=DATA_PROCESSING_IMAGE,
                command=[],
                args=get_arguments(args, "data_processing"),
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def data_processing_pipeline():
            data_processing().set_cpu_limit("16")

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_processing_pipeline, package_path="data_processing.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-data-processing-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_processing.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.submit(service_account=GCS_SERVICE_ACCOUNT)


    #if args.model_training:
    #    aip.init(project=GCP_PROJECT, location=GCS_REGION, staging_bucket=BUCKET_URI)

    #    job_id = generate_uuid()
    #    DISPLAY_NAME = "dog_breeds_" + job_id

    #    job = aip.CustomPythonPackageTrainingJob(
    #        display_name=DISPLAY_NAME,
    #        python_package_gcs_uri=f"{GCS_PACKAGE_URI}/dog-breed-trainer.tar.gz",
    #        python_module_name="trainer.task",
    #        container_uri=MODEL_TRAIN_IMAGE,
    #        project=GCP_PROJECT,
    #    )

    #    CMDARGS = ["--epochs=15", "--batch_size=16"]
    #    MODEL_DIR = GCS_PACKAGE_URI
    #    TRAIN_COMPUTE = "n1-standard-4"
    #    TRAIN_GPU = "NVIDIA_TESLA_T4"
    #    TRAIN_NGPU = 1

        # Run the training job on Vertex AI
    #    job.submit(
    #        model_display_name=None,
    #        args=CMDARGS,
    #        replica_count=1,
    #        machine_type=TRAIN_COMPUTE,
    #        accelerator_type=TRAIN_GPU,
    #        accelerator_count=TRAIN_NGPU,
    #        base_output_dir=MODEL_DIR,
    #        sync=False,
    #    )

    if args.model_training:
        print("Model Training")

        # Define a Pipeline
        @dsl.pipeline
        def model_training_pipeline():
            model_training(
                project=GCP_PROJECT,
                location=GCS_REGION,
                staging_bucket=GCS_PACKAGE_URI,
                bucket_name=GCS_BUCKET_NAME,
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            model_training_pipeline, package_path="model_training.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-app-model-training-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="model_training.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)



    if args.model_deploy:
        print("Model Deploy")

        # Define a Pipeline
        @dsl.pipeline
        def model_deploy_pipeline():
            model_deploy(
                bucket_name=GCS_BUCKET_NAME,
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            model_deploy_pipeline, package_path="model_deploy.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-app-model-deploy-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="model_deploy.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.pipeline:
        @dsl.container_component
        def data_extraction():
            container_spec = dsl.ContainerSpec(
                image=DATA_EXTRACTION_IMAGE,
                command=[],
                args=get_arguments(args, "data_extraction"),
            )
            return container_spec

        @dsl.container_component
        def data_transformation():
            container_spec = dsl.ContainerSpec(
                image=DATA_TRANSFORMATION_IMAGE,
                command=[],
                args=get_arguments(args, "data_transformation"),
            )
            return container_spec

        @dsl.container_component
        def data_preprocessing():
            container_spec = dsl.ContainerSpec(
                image=DATA_PREPROCESSING_IMAGE,
                command=[],
                args=get_arguments(args, "data_preprocessing"),
            )
            return container_spec

        @dsl.container_component
        def data_processing():
            container_spec = dsl.ContainerSpec(
                image=DATA_PROCESSING_IMAGE,
                command=[],
                args=get_arguments(args, "data_processing"),
            )
            return container_spec


        # Define a Pipeline
        @dsl.pipeline
        def data_pipeline():
            # Data Collector
            data_extraction_task = (
                data_extraction()
                .set_display_name("Data Extraction")
                .set_cpu_limit("32")
            )

            # Data Processor
            data_transformation_task = (
                data_transformation()
                .set_display_name("Data Transformation")
                .set_cpu_limit("32")
                .after(data_extraction_task)
            )

            # Data Preprocessing
            data_preprocessing_task = (
                data_preprocessing()
                .set_display_name("Data Preprocessing")
                .set_cpu_limit("32")
                .after(data_transformation_task)
            )

            # Data Processing
            data_processing_task = (
                data_processing()
                .set_display_name("Data Processing")
                .set_cpu_limit("16")
                .after(data_preprocessing_task)
            )
            # Model Training
            model_training_task = (
                model_training(
                    project=GCP_PROJECT,
                    location=GCS_REGION,
                    staging_bucket=GCS_PACKAGE_URI,
                    bucket_name=GCS_BUCKET_NAME,
                    epochs=15,
                    batch_size=16,
                    model_name="mobilenetv2",
                    train_base=False,
                )
                .set_display_name("Model Training")
                .after(data_processing_task)
            )

            # Model Deployment
            model_deploy_task = (
                model_deploy(
                    bucket_name=GCS_BUCKET_NAME,
                )
                .set_display_name("Model Deploy")
                .after(model_training_task)
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(data_pipeline, package_path="pipeline.yaml")

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-data-pipeline-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="pipeline.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.submit(service_account=GCS_SERVICE_ACCOUNT)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "-x",
        "--data_extraction",
        action="store_true",
        help="Extract data from sources to GCS bucket",
    )

    parser.add_argument(
        "-t",
        "--data_transformation",
        action="store_true",
        help="Transform images and annotations for breed data",
    )

    parser.add_argument(
        "-p",
        "--data_preprocessing",
        action="store_true",
        help="Resize images and create training splits for breed data",
    )

    parser.add_argument(
        "-s",
        "--data_processing",
        action="store_true",
        help="Convert images and annotations to TFRecord files",
    )

    parser.add_argument(
        "-m",
        "--model_training",
        action="store_true",
        help="Train model using TFRecord files",
    )
    parser.add_argument(
        "-d",
        "--model_deploy",
        action="store_true",
        help="Run just Model Deployment",
    )
    parser.add_argument(
        "-w",
        "--pipeline",
        action="store_true",
        help="Run full Data Pipeline and",
    )

    # Optional image size (otherwise default=224)
    parser.add_argument(
        "-i",
        "--image_size",
        nargs='?',
        type=int,
        help="Optional output image size",
    )

    parser.add_argument(
        "-c",
        "--cropped",
        action="store_true",
        help="Create dataset using cropped breed images",
    )

    parser.add_argument(
        "-u",
        "--uncropped",
        action="store_true",
        help="Create dataset using uncropped breed images",
    )

    parser.add_argument(
        "-n",
        "--individual",
        action="store_true",
        help="Create dataset using individual dog images",
    )

    args = parser.parse_args()

    main(args)