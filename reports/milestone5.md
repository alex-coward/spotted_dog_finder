
# AC215 - SPOTTED! - Main Branch


**Team Members**  
Alex Coward, Olga Leushina

**Group Name**  
Spotted!

**Project**  

Our project seeks to identify lost dogs from images uploaded by spotters and find potential matches among dogs reported missing by owners. Initially we aimed to use both a breed classifier and similarty search on image vector embeddings to accomplish the task. As our work has developed, we have decided to focus on the image vector embedding similarity search method.
 
The machine learning component consists of passing uploaded images of lost or found dogs through an object detection model to then crop the image based on predicted bounding boxes. We then pass the cropped image through a vision transformer model to generate vector embeddings. The vector embedding is then checked against an FAISS index of vector embeddings from previously uploaded images of either found or lost dogs, respectievley, to determine if there is a match.


### Milestone 4 Additional work

Our entire data pipeline now runs serverless using Vertex AI. Due to team member dropping off we had to re-build everything in the pipeline to accomodate for a change in buckets structure and create new images in another Docker Hub account. 
To accomodate this change, we refactored our entire data pipeline so each step now runs in its own container. There's a separate container that can be run to train the model (serverless) and one more to deploy the model - they are all ready and running. 
Vertex AI pipeline now includes steps for basic training of the model based on the inputs and also for deployment so that there's a Vertex AI endpoint available to be called.

Here is the screenshot of the full Vertex AI pipeline:

![Alt text][def]

![Alt text][def2]


## Milestone 5

After completions of building a robust ML Pipeline in our previous milestone we have built a backend api service and frontend app. This will be our user-facing application that ties together the various components built in previous milestones.

**Application Design**

Before we start implementing the app we built a detailed design document outlining the application’s architecture. We built a Solution Architecture abd Technical Architecture to ensure all our components work together.


Here is our Solution Architecture:
![Solution architecture][def3]


Here is our Technical Architecture:
![Technical architecture][def4]


### Deployment has been done and documented in src/deployment/README.md  - this is where we have screenshots of manual deployment to VM, ansible-playbooks deployment and k8s cluster deployment that we need for scalability ###



**Backend API**

We built a backend api service using fast API to expose model functionality to the frontend. The api does 2 broad tasks. It handles all the processing of uplaoded images through the object detection model, the vision transformer model, and the FAISS index for similarity search. The api also saves to persistent disk all uploaded images, their generated vector embeddings, and user uploaded contact info for use in future matches. 

The api then returns to the front end any images and contact info for matches for up to the top 5 matching images with cosine similarity of at least 0.7.

![Api service](reports/screenshots/spotted_api_screenshot.png)



**Frontend**

A user friendly React app was built to allow for submission of lost or found dog image and contact info and to receive images of possible matches for the dog along with contact info of the uploader of the relevant dog image.  

Here are some screenshots of the app

App Home
![App Home](reports/screenshots/spotted_app_screenshots/Spotted_Screenshot_home.png)

Dog Matching Page
![Dog Matching](reports/screenshots/spotted_app_screenshots/spotted_screenshot_image_matching.png)

Data Entry for Lost Dog
![Lost Dog Data Entry](reports/screenshots/spotted_app_screenshots/spotted_screenshot_lost_dog_data_entry.png)

No Matches for Lost Dog
![Lost Dog No Match](reports/screenshots/spotted_app_screenshots/spotted_screenshot_lost_dog_no_matches.png)

Data Entry for Found Dog
![Found Dog Data Entry](reports/screenshots/spotted_app_screenshots/spotted_screenshot_found_dog_data_entry.png)

Match for Lost Dog
![Lost Dog No Match](reports/screenshots/spotted_app_screenshots/spotted_screenshot_found_dog_match.png)





#### Code Structure

Project Organization
------------

    .
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
    │   ├── frontend-application     <- frontend application code and files
    │   │   ├── api-service     <- Code for App backend APIs
    │   │   │   ├── api
    │   │   │   │   ├── ...
    │   │   ├── docker-entrypoint.sh
    │   │   ├── docker-shell.sh
    │   │   ├── docker-shell.bat
    │   │   ├── Dockerfile
    │   │   ├── Pipfile
    │   │   ├── Pipfile.lock
    │   ├── deployment          <- Code for App deployment to GCP
    │   │   ├── nginx-conf
    │   │   │   ├── ...    
    │   │   ├── .docker-tag
    │   │   ├── .gitignore
    │   │   ├── deploy-create-instance.yml
    │   │   ├── deploy-docker-images.yml
    │   │   ├── deploy-k8s-cluster.yml
    │   │   ├── deploy-k8s-tic-tac-toe.yml
    │   │   ├── deploy-provision-instance.yml
    │   │   ├── deploy-setup-containers.yml
    │   │   ├── deploy-setup-webserver.yml
    │   │   ├── docker-entrypoint.sh
    │   │   ├── docker-shell.bat
    │   │   ├── docker-shell.sh
    │   │   ├── Dockerfile
    │   │   ├── inventory.yml
    │   ├── frontend-react            <- Code for App frontend
    │   │   ├── public
    │   │   │   ├── ...  
    │   │   ├── src
    │   │   │   ├── ...  
    │   │   ├── env.development
    │   │   ├── env.production
    │   │   ├── .gitignore
    │   │   ├── docker-shell.bat
    │   │   ├── docker-shell.sh
    │   │   ├── Dockerfile
    │   │   ├── Dockerfile.dev
    │   │   ├── package.json
    │   │   └── yarn.lock
    │   ├── model-training     <- Model training, evaluation, and prediction code
    │   │   ├── ...
    │   └── model-deployment       <- Model deployment
    │   │   ├── ...
    │   └── serverless_training_distillation       <- container for running serverless training with distillation model
    │   │   ├── ...
    │   ├── workflow           <- Scripts for automating data collection, preprocessing, modeling
    │   │   ├── ...

    
--------


[def]: reports/screenshots/Vertex_AI_pipeline_full_1.png
[def2]: reports/screenshots/Vertex_AI_pipeline_full_2.png
[def3]: reports/screenshots/solution_architecture.png
[def4]: reports/screenshots/technical_architecture.png

