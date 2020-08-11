source config_env.sh

zip sm_project_exp.zip generate_markets.py bidders.py experiments_pyspark.py market_constituents.py market.py
gsutil cp sm_project_exp.zip gs://${BUCKET_NAME}/
gsutil cp generate_markets.py gs://${BUCKET_NAME}/
gsutil cp experiments_pyspark.py gs://${BUCKET_NAME}/