source config_env.sh
gcloud dataproc jobs submit pyspark gs://${BUCKET_NAME}/generate_markets.py \
    --cluster=${CLUSTER} \
    --project=${PROJECT} \
    --region=${REGION} \
    --py-files gs://${BUCKET_NAME}/sm_project_exp.zip \
    -- $1 gs://${BUCKET_NAME}/non_iso_markets/ gs://${BUCKET_NAME}/ 3 4