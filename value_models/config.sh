# Create a new compute-optimized instance
gcloud compute instances create experiments --machine-type=c2-standard-4 --zone=us-east1-b
# ssh into the instance
gcloud compute ssh experiments
# install screen
sudo apt-get install screen

# After experiments are done and all files are moved, delete instance.
gcloud compute instances delete experiments
