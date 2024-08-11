# AC215 - SPOTTED! - Main Branch


**Team Members**  
Alex Coward, Olga Leushina

**Group Name**  
Spotted!

### Presentation  Video
* https://youtu.be/zoyyRo06kFg

### Blog Post Link
*  https://medium.com/institute-for-applied-computational-science/spotted-an-app-to-help-lost-dogs-get-home-f89d796fe0a4
---



### Project  

The purpose of the application that we created is to identify lost dogs from images uploaded by spotters and find potential matches among dogs reported missing by owners. Initially we aimed to use both a breed classifier and similarty search on image vector embeddings to accomplish the task. As our work has developed, we have decided to focus on the image vector embedding similarity search method.
 
The machine learning component consists of passing uploaded images of lost or found dogs through an object detection model to then crop the image based on predicted bounding boxes. We then pass the cropped image through a vision transformer model to generate vector embeddings. The vector embedding is then checked against an FAISS index of vector embeddings from previously uploaded images of either found or lost dogs, respectievley, to determine if there is a match.


### Milestone 6

After building the frontend app and deployment to k8s cluster we added CI/CD automatically-triggered deployments using GitHub Actions. We also worked on scalability and model upload optimization.
As a result, we have an application that is automatically scalable and can be re-deployed automatically if a comment is added to git push commit.

Our yaml files can be found under `.github/workflows`

Additionally, there is a CI/CD pipeline that can be run to download and process data and run training and deployment to Vertex AI.


* Application running on K8s:
![App running k8s][def5]

* Full Data processing-training-deployment pipeline running on Vertex AI:
![Pipeline running Vertex AI][def6]



***A Very Detailed documentation of MLOps part of the project can be found in src/deployment/README.md - points from 7 down are related to Milestone 6 work.***



### React Application


Here are some screenshots of our app:



* Lost dog (no match found)
![Lost dog][def1]



* Match found!

![Found dog][def2]




### Solution architecture and technical architecture

We have updated the architecture images with added CI/CD part using GitHub Actions

* Final Solution architecture

![Solution architecture updated][def3]

* Final Technical architecture
![Technical architecture updated][def4]



### Code Structure

Project Organization
------------
    ├── .github/workflows - added in Milestone 6
    │   └── ci-cd.yml    
    ├── data # NO DATA UPLOADED
    │   ├── interim          <- Intermediate preprocessed data
    │   │   ├── readme.md
    │   ├── processed        <- Final dataset files for modeling
    │   │   ├── all-data_stanford.csv
    │   │   ├── austin_pets_alive_tfrecords_1_of_2.png
    │   │   ├── austin_pets_alive_tfrecords_2_of_2.png
    │   │   └── readme.md
    │   └── raw              <- Original immutable input data
    │   │   ├── austin_pets_alive_original_1_of_2.png
    │   │   ├── austin_pets_alive_original_2_of_2.png
    │   │   └── readme.md  
    │   ├── README.md
    ├── notebooks            <- Jupyter notebooks for EDA and model testing
    │   ├── matching            <- Jupyter notebooks for matching model
    │   │   ├── notebooks
    │   │   │   ├── faiss.ipynb
    │   │   │   ├── image_encoder.ipynb
    │   │   │   ├── object_detection_plus_embeddings.ipynb      
    │   │   ├── src
    │   │   │   ├── faiss_tutorial.py
    │   │   │   ├── vit_mae_encoder.py
    │   │   ├── .gitignore
    │   │   ├── Dockerfile
    │   │   ├── installation_notes.txt
    │   │   ├── LICENSE 
    │   │   ├── README.md     
    │   ├── breed_data_eda.ipynb
    │   ├── classifier_with_tfr.ipynb
    │   ├── models_train_wandb_on_tfrecords_processed.ipynb
    │   ├── models_train_wandb.ipynb
    │   └── object_detection.ipynb
    ├── presentations        <- Folder containing our midterm presentation
    │   └── midterm.pdf
    ├── references           <- Reference papers
    │   ├── Mao - Dog Recognition usind CNNs (2023).pdf
    │   ├── Tsinghua Dogs Dataset.pdf
    │   ├── Yifeng Lan - Pet Finding(2022).pdf
    │   └── Zaman - Deep Learning Pet ID (2023).pdf
    ├── reports              <- Folder containing your milestone markdown submissions
    |   ├── screenshots
    │   ├── milestone2.md
    │   └── milestone3.md
    │   └── milestone4.md
    │   └── milestone5.md
    ├── src                    <- Source code and Dockerfiles for data processing and modeling
    │   ├── api-service     <- Code for App backend APIs
    │   │   │   ├── api
    │   │   │   │   ├── ...
    │   ├── data_extraction     <- Data Extraction container code and files
    │   │   ├── ...
    │   ├── data_preprocessing     <- Data Preprocessing container code and files
    │   │   ├── ...
    │   ├── data_processing     <- Data Processing container code and files
    │   │   ├── ...
    │   ├── data_transformation     <- Data Transformation container code and files
    │   │   ├── ...
    │   ├── data_versioning     <- Data Versioning container code and files
    │   │   ├── ...
    │   ├── deployment          <- Code for App deployment to GCP - updated Milestone 6
    │   │   ├── readme_images
    │   │   │   ├── ...    
    │   │   ├── .docker-tag
    │   │   ├── .gitignore
    │   │   ├── data_extraction.yaml
    │   │   ├── data_preprocessing.yaml
    │   │   ├── data_processing.yaml
    │   │   ├── data_transformation.yaml
    │   │   ├── deploy-app.sh
    │   │   ├── deploy-docker-images.yml
    │   │   ├── deploy-k8s-cluster.yml
    │   │   ├── docker-entrypoint.sh
    │   │   ├── docker-shell.bat
    │   │   ├── Dockerfile
    │   │   ├── inventory-prod.yml
    │   │   ├── inventory.yml
    │   │   ├── model_deployment.yaml
    │   │   ├── model_training.yaml
    │   │   ├── model.py
    │   │   ├── Pipfile
    │   │   ├── Pipfile.lock
    │   │   ├── README.md
    │   │   ├── run-data-extraction.sh
    │   │   ├── run-data-preprocessing.sh
    │   │   ├── run-data-processing.sh
    │   │   ├── run-data-transformation.sh
    │   │   ├── run-ml-pipeline.sh
    │   │   ├── run-model-deployment.sh
    │   │   ├── run-model-training.sh
    │   │   ├── run-data-extraction.sh
    │   │   ├── update-k8s-cluster.yml
    │   ├── frontend-react            <- Code for App frontend
    │   │   ├── node_modules
    │   │   │   ├── ... 
    │   │   ├── public
    │   │   │   ├── ...  
    │   │   ├── src
    │   │   │   ├── ...  
    │   │   ├── env.development
    │   │   ├── env.production
    │   │   ├── .eslintcache
    │   │   ├── .gitignore
    │   │   ├── docker-shell.bat
    │   │   ├── docker-shell.sh
    │   │   ├── Dockerfile
    │   │   ├── Dockerfile.dev
    │   │   ├── package.json
    │   │   ├── README.md    
    │   │   └── yarn.lock
    │   ├── model-training     <- Model training, evaluation, and prediction code
    │   │   ├── ...
    │   └── model-deployment       <- Model deployment
    │   │   ├── ...
    │   └── serverless_training_distillation       <- container for running serverless training with distillation model
    │   │   ├── ...
    │   ├── workflow           <- Scripts for automating data collection, preprocessing, modeling
    │   │   ├── ...
    │   ├── README.md
    
--------

[def1]: reports/screenshots/lost_dog.png
[def2]: reports/screenshots/found_dog.png
[def3]: reports/screenshots/solution_architecture_final.png
[def4]: reports/screenshots/technical_architecture_final.png
[def5]: reports/screenshots/app-running-k8s.png
[def6]: reports/screenshots/image-43.png
