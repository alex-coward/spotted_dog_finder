#ansible-playbook deploy-docker-images.yml -i inventory.yml
#ansible-playbook update-k8s-cluster.yml -i inventory-prod.yml
ansible-playbook deploy-k8s-cluster.yml -i inventory-prod.yml --extra-vars cluster_state=present