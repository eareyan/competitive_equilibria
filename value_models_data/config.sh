# Create a new compute-optimized instance
gcloud compute instances create experiments2 --machine-type=c2-standard-4 --zone=us-east1-b
# ssh into the instance
gcloud compute ssh experiments2
# After experiments are done and all files are moved, delete instance.
# gcloud compute instances delete experiments1
