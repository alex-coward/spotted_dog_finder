# Data Versioning Container

The data versioning container is intended to run on a GCP VM with GCS buckets mounted via [GCS-Fuse](https://cloud.google.com/storage/docs/gcs-fuse) to record the current state of our data and track via our Git repo.

## Initial Setup of GCP VM
* Create a service account with permissions for writing to cloud buckets (roles/storage.objectAdmin)
* Create a VM Instance from [GCP](https://console.cloud.google.com/compute/instances) 
  * Set region to the same as GCS for improved performance (defined in `setup.sh`)
  * Under "Identity and API Access" attach your service account
* SSH into the newly created instance


## Starting the Container

Download and install Docker:
> `curl -fsSL https://get.docker.com -o get-docker.sh`  
> `sudo sh get-docker.sh`  

Install Git:  
> `sudo apt-get install git`

Create Git SSH Key:   
> `ssh-keygen -t ed25519 -C "email@g.harvard.edu"`  (Press enter for all 3 prompts)  
> `cat /home/<username>/.ssh/id_ed25519.pub`  (To display public key)  

Add your public SSH key to your GitHub account:
* Navigate to GitHub.com and log in
* Go to Settings (under your profile picture on the top right)
* Select "SSH and GPG Keys" from the left menu
* Click the "New SSH Key" button
* Enter a title and paste in your public key from the VM
* Click the "Add SSH Key" button (you may need to enter your GitHub password)  

Verify your SSH Key is working:
> `ssh -T git@github.com`

Clone the repository:
> `git clone -b <branchname> git@github.com:<github_account>/AC215_spotted`

Copy the following files from `AC215_spotted/src/data_versioning` to `AC215_spotted/`:
* Dockerfile
* docker-entrypoint.sh
* docker-shell.sh
* Pipfile
* Pipfile.lock

Update account and user settings:  
* `docker-entrypoint.sh` - Update GitHub account information
* `docker-shell.sh` - Update GCP project and service account information

Build and run the Docker image:  
> `sudo sh docker-shell.sh`  
> `sudo docker run --rm --privileged --name dvc-docker-image -it dvc-docker-image`


## Data Version Control

Create another SSH key inside the Docker container following same instructions as above and add it to your GitHub account. Verify your GitHub connection:  
> `ssh -T git@github.com`

Create local directory and mount to GCS bucket:
> `cd ./src/data_versioning`  
> `mkdir dvc-data`  
> `gcsfuse <bucket_name> dvc-data/`

If you want to mount to a directory within a bucket:
> `gcsfuse --only-dir dir1/dir2 <bucket_name> dvc-data/`

Create DVC repository:
> `dvc init --subdir`  
> `dvc remote add -d dvc-data gs://<bucket_name>/dvc_store`  
> `dvc add dvc-data`  
> `dvc push`

Remove `src/data_versioning` files copied into the repo root above.  

Update Git repository:
> `git status`  
> `git add .`  
> `git commit -m 'Commit message'`  
> `git tag -a 'dataset_v1' -m 'tag dataset`  
> `git push --atomic origin <branch_name> dataset_v1`  

The tag should now appear in the GitHub repo and files should be saved in the `dvc_store` folder of your GCS bucket.
