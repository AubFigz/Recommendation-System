AI-Powered Recommendation System with AWS, Airflow, and Flask API
This project is an AI-powered recommendation system that utilizes user behavior data to provide real-time item recommendations. The system integrates multiple components, including data ingestion, machine learning model training, API deployment, model monitoring, and orchestration using AWS Lambda, Airflow, and Flask.

The entire system covers end-to-end data ingestion, model training, real-time recommendation retrieval via API, automatic updating using Airflow, and monitoring using AWS CloudWatch.

Project Overview
This system is designed to:

Simulate and ingest user behavior data: Simulated data is uploaded to cloud storage systems like AWS S3 and Azure Blob Storage.
Train a recommendation model: Using Non-negative Matrix Factorization (NMF), a model is trained to recommend items based on user interactions.
Serve recommendations: Via a Flask API that allows users to request recommendations for specific users.
Automate model updates: Using Apache Airflow, the model is updated daily, and any outdated models are backed up or cleaned.
Monitor the system: Important metrics like model latency and accuracy are logged using AWS CloudWatch for monitoring and diagnostics.

Files and Modules
1. data_ingestion.py
This script simulates user behavior data (user interactions with items), anonymizes it, and uploads it to either Azure Blob Storage or AWS S3.

Functions:
simulate_behavior_data: Simulates user interactions with random items.
anonymize_data: Hashes the user IDs for privacy.
upload_to_azure: Uploads the anonymized data to Azure Blob Storage.
upload_to_s3: Uploads the anonymized data to AWS S3.

Key Points:
It supports both Azure and AWS cloud uploads.
Data anonymization is performed for privacy protection.

2. recommendation_model.py
This script loads the ingested user behavior data, constructs a user-item interaction matrix, and trains an NMF recommendation model.

Functions:
load_data: Loads data from a CSV file.
build_interaction_matrix: Converts user-item interactions into a matrix for model training.
train_nmf_model: Trains the NMF model to extract latent user and item features.
get_recommendations: Generates recommendations for a given user based on the trained model.

Key Points:
The trained model produces user and item features, which are used to generate top-N recommendations.
Supports lazy loading and error handling for robust operations.

3. api.py
This script provides a Flask-based API that serves recommendations for users by interfacing with the NMF recommendation model.

Functions:
recommend: API endpoint that accepts POST requests with user_id and returns recommendations for that user.

Key Points:
The Flask API uses lazy loading of models to optimize performance.
Validation is performed on the incoming request (ensuring valid user_id).
It provides real-time recommendations based on the trained NMF model.

4. deploy_lambda.py
This script is designed for deployment to AWS Lambda. It exposes a serverless recommendation API that retrieves recommendations based on the trained model.

Functions:
lambda_handler: AWS Lambda function handler that accepts user input (user_id) and returns recommendations.

Key Points:
It supports lazy loading of the model to optimize AWS Lambda cold start performance.
Environment variables can be used to control model settings and data paths.
Input validation is included to ensure correct requests.

5. monitoring.py
This script logs performance metrics (latency, model accuracy) to AWS CloudWatch.

Functions:
log_metric: Logs custom metrics to AWS CloudWatch.
log_latency: Logs model prediction latency.
log_model_accuracy: Logs model accuracy.

Key Points:
AWS CloudWatch is used to monitor key performance metrics like model latency and accuracy.
Provides detailed logs for monitoring and debugging.

6. airflow_dag.py
This is the Apache Airflow DAG responsible for automating the daily update of the recommendation model. It validates the updated model, backs up old models to AWS S3, and cleans up outdated models.

Tasks:
update_model_task: Updates the recommendation model.
validate_model_task: Validates the model using reconstruction error.
backup_model_task: Backs up the updated model and interaction matrix to AWS S3.
cleanup_old_models_task: Cleans up old models from storage.
notify_success_task: Sends a notification email upon successful model update.

Key Points:
The DAG is scheduled to run daily and handles model updates, validation, and backup.
It integrates directly with AWS services for model backup and notifications.

Environment Variables
The following environment variables should be set for the system to function properly:

Data Ingestion:
AZURE_CONNECTION_STRING: Azure Blob Storage connection string for data upload.
AWS_ACCESS_KEY_ID: AWS access key for S3 upload.
AWS_SECRET_ACCESS_KEY: AWS secret key for S3 upload.
AWS_REGION: AWS region for S3 uploads.

Recommendation Model and API:
BEHAVIOR_DATA_PATH: Path to the user behavior data CSV file.
NMF_COMPONENTS: Number of components for the NMF model (default: 15).

Airflow:
S3_BUCKET_NAME: Name of the AWS S3 bucket for model backups.
AIRFLOW_EMAIL: Email address for Airflow notifications.

AWS Lambda:
BEHAVIOR_DATA_PATH: Path to the user behavior data CSV file (can be in S3).
NMF_COMPONENTS: Number of components for the NMF model.
AWS_REGION: AWS region for the Lambda function.

Monitoring:
CLOUDWATCH_NAMESPACE: AWS CloudWatch namespace for logging metrics (default: RecommendationSystem).
AWS_REGION: AWS region for CloudWatch logs.

How to Deploy and Run
1. Data Ingestion
To ingest and anonymize behavior data and upload to cloud storage:
python data_ingestion.py

2. Running the Flask API
Install dependencies:
pip install flask pandas numpy scikit-learn

Run the Flask API:
python api.py

3. Deploying to AWS Lambda
Package the deploy_lambda.py script with dependencies (e.g., pandas, numpy, scikit-learn) and deploy to AWS Lambda.
Set the necessary environment variables in the AWS Lambda configuration.

4. Setting Up Airflow DAG
Place the airflow_dag.py file in your Airflow dags folder.
Set the necessary environment variables for AWS S3 and model configuration.

Start Airflow:
airflow scheduler
airflow webserver

5. Monitoring with CloudWatch
Run the monitoring.py script to log metrics:
python monitoring.py

Prerequisites and External Requirements
1. Cloud Resources:
AWS S3: For storing ingested data, model backups, and interaction matrices.
AWS CloudWatch: For logging metrics related to model performance (latency, accuracy).
AWS Lambda: For deploying the serverless recommendation API.
Azure Blob Storage: For optionally storing ingested user behavior data.
Apache Airflow: For automating model updates and backups.

2. Python Dependencies:
Ensure that the following libraries are installed:
flask
pandas
numpy
scikit-learn
boto3
azure-storage-blob
airflow
