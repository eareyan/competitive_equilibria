source config_env.sh

gcloud dataproc clusters create ${CLUSTER} \
  --project=${PROJECT} \
  --region=${REGION} \
  --max-idle=30m \
  --image-version=1.5 \
  --master-machine-type=n1-standard-2 \
  --worker-machine-type=n1-standard-2 \
  --num-workers=3 \
  --properties spark:spark.yarn.executor.memoryOverhead=2G \
  --metadata 'PIP_PACKAGES=PuLP==2.2 prettytable==0.7.2 PTable==0.9.2' \
  --initialization-actions gs://goog-dataproc-initialization-actions-${REGION}/python/pip-install.sh
