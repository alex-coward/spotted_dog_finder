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
from model import model_training, model_deployment


GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/root"
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]
GCS_PACKAGE_URI = os.environ["GCS_PACKAGE_URI"]
GCP_REGION = os.environ["GCP_REGION"]

# Read the docker tag file
with open(".docker-tag") as f:
    tag = f.read()

tag = tag.strip()

print("Tag>>", tag, "<<")

# Code to run the latest images
#DATA_EXTRACTION_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-data-extraction:{tag}"
#DATA_TRANSFORMATION_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-data-transformation:{tag}"
#DATA_PREPROCESSING_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-data-preprocessing:{tag}"
#DATA_PROCESSING_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-data-processing:{tag}"
#MODEL_TRAINING_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-model-training:{tag}"
#MODEL_DEPLOYMENT_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-model-deployment:{tag}"

# Code to run the images already in GCS
DATA_EXTRACTION_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-data-extraction:20231206192245"
DATA_TRANSFORMATION_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-data-transformation:20231206192245"
DATA_PREPROCESSING_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-data-preprocessing:20231206192245"
DATA_PROCESSING_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-data-processing:20231206195811"
MODEL_TRAINING_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-model-training:20231211220048"
MODEL_DEPLOYMENT_IMAGE = f"gcr.io/{GCP_PROJECT}/spotted-model-deployment:20231211223006"

def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def main(args=None):
    print("CLI Arguments:", args)

    if args.data_extraction:
        # Define a Container Component
        @dsl.container_component
        def data_extraction():
            container_spec = dsl.ContainerSpec(
                image=DATA_EXTRACTION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--all",
                    "--breed",
                    "--individual",
                ],
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
        DISPLAY_NAME = "spotted-app-data-extraction-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_extraction.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.data_transformation:
        print("Data Transformation")

        # Define a Container Component for data transformation
        @dsl.container_component
        def data_transformation():
            container_spec = dsl.ContainerSpec(
                image=DATA_TRANSFORMATION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                ],
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def data_transformation_pipeline():
            data_transformation()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_transformation_pipeline, package_path="data_transformation.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-app-data-transformation-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_transformation.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.data_preprocessing:
        print("Data Preprocessing")

        # Define a Container Component
        @dsl.container_component
        def data_preprocessing():
            container_spec = dsl.ContainerSpec(
                image=DATA_PREPROCESSING_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--cropped",
                    "--uncropped",
                    "--image_size",
                ],
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def data_preprocessing_pipeline():
            data_preprocessing()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_preprocessing_pipeline, package_path="data_preprocessing.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-app-data-preprocessing-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_preprocessing.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.data_processing:
        print("Data Processing")

        # Define a Container Component for data processor
        @dsl.container_component
        def data_processing():
            container_spec = dsl.ContainerSpec(
                image=DATA_PROCESSING_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--cropped",
                    "--uncropped",
                    "--image_size",
                ],
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def data_processing_pipeline():
            data_processing()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_processing_pipeline, package_path="data_processing.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-app-data-processing-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_processing.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.model_training:
        print("Model Training")

        # Define a Pipeline
        @dsl.pipeline
        def model_training_pipeline():
            model_training(
                project=GCP_PROJECT,
                location=GCP_REGION,
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

    if args.model_deployment:
        print("Model Deployment")

        # Define a Pipeline
        @dsl.pipeline
        def model_deployment_pipeline():
            model_deployment(
                bucket_name=GCS_BUCKET_NAME,
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            model_deployment_pipeline, package_path="model_deployment.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-app-model-deployment-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="model_deployment.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.pipeline:
        # Define a Container Component for data extraction
        @dsl.container_component
        def data_extraction():
            container_spec = dsl.ContainerSpec(
                image=DATA_EXTRACTION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--all",
                    "--breed",
                    "--individual",
                ],
            )
            return container_spec

        # Define a Container Component for data transformation
        @dsl.container_component
        def data_transformation():
            container_spec = dsl.ContainerSpec(
                image=DATA_TRANSFORMATION_IMAGE,
                command=[],
                args=[
                    "cli.py",
                ],
            )
            return container_spec
        
        # Define a Container Component for data preprocessing
        @dsl.container_component
        def data_preprocessing():
            container_spec = dsl.ContainerSpec(
                image=DATA_PREPROCESSING_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--cropped",
                    "--uncropped",
                    "--image_size",
                ],
            )
            return container_spec
        
        # Define a Container Component for data processing
        @dsl.container_component
        def data_processing():
            container_spec = dsl.ContainerSpec(
                image=DATA_PROCESSING_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--cropped",
                    "--uncropped",
                    "--image_size",
                ],
            )
            return container_spec
        
        # Define a Container Component for model training
        @dsl.container_component
        def model_training():
            container_spec = dsl.ContainerSpec(
                image=MODEL_TRAINING_IMAGE,
                command=[],
                args=[
                    "cli.py",
                ],
            )
            return container_spec

        # Define a Container Component for model deployment
        @dsl.container_component
        def model_training():
            container_spec = dsl.ContainerSpec(
                image=MODEL_DEPLOYMENT_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--upload",
                    "--deploy",
                    "--predict",
                    "--test",
                ],
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def ml_pipeline():
            # Data Extraction
            data_extraction_task = (
                data_extraction()
                .set_display_name("Data Extraction")
                .set_cpu_limit("500m")
                .set_memory_limit("2G")
            )
            # Data Transformation
            data_transformation_task = (
                data_transformation()
                .set_display_name("Data Transformation")
                .after(data_extraction_task)
            )
            # Data Preprocessing
            data_preprocessing_task = (
                data_preprocessing()
                .set_display_name("Data Preprocessing")
                .after(data_transformation_task)
            )
            # Data Processing
            data_processing_task = (
                data_processing()
                .set_display_name("Data Processing")
                .after(data_preprocessing_task)
            )
            # Model Training
            model_training_task = (
                model_training()
                .set_display_name("Model Training")
                .after(data_processing_task)
            )
            # Model Deployment
            model_deployment_task = (
                model_deployment()
                .set_display_name("Model Deployment")
                .after(model_training_task)
            )
            
        # Build yaml file for pipeline
        compiler.Compiler().compile(ml_pipeline, package_path="pipeline.yaml")

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "spotted-app-pipeline-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="pipeline.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "-e",
        "--data_extraction",
        action="store_true",
        help="Run just the Data Extraction",
    )
    parser.add_argument(
        "-t",
        "--data_transformation",
        action="store_true",
        help="Run just the Data Transformation",
    )
    parser.add_argument(
        "-p",
        "--data_preprocessing",
        action="store_true",
        help="Run just the Data Preprocessing",
    )
    parser.add_argument(
        "-c",
        "--data_processing",
        action="store_true",
        help="Run just the Data Processing",
    )
    parser.add_argument(
        "-m",
        "--model_training",
        action="store_true",
        help="Run just Model Training",
    )
    parser.add_argument(
        "-d",
        "--model_deployment",
        action="store_true",
        help="Run just Model Deployment",
    )
    parser.add_argument(
        "-w",
        "--pipeline",
        action="store_true",
        help="Spotted App Pipeline",
    )

    args = parser.parse_args()

    main(args)
