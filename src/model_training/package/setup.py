from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ["wandb", "tensorflow-hub", "transformers"]

setup(
    name="vit-model-trainer",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="VIT Model Trainer Application",
)
