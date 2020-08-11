source config_env.sh
gcloud dataproc jobs submit pyspark gs://${BUCKET_NAME}/experiments_pyspark.py \
    --cluster=${CLUSTER} \
    --project=${PROJECT} \
    --region=${REGION} \
    --py-files gs://${BUCKET_NAME}/sm_project_exp.zip \
    -- $1 gs://${BUCKET_NAME}/ gs://${BUCKET_NAME}/experiment_results/ 3 4