import os
import logging
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta
from recommendation_model import load_data, build_interaction_matrix, train_nmf_model
import boto3

# Initialize logging for the Airflow DAG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 3,  # Retry three times in case of failure
    'retry_delay': timedelta(minutes=5),  # Wait 5 minutes between retries
    'email_on_failure': False,
    'email_on_retry': False
}

# Define the DAG with a daily schedule
dag = DAG(
    'recommendation_system_update_extended',
    default_args=default_args,
    description='DAG for updating and validating the recommendation model, with backup and notifications',
    schedule_interval='@daily',  # Run the task every day
    catchup=False  # Do not run past scheduled jobs if Airflow was offline
)


# Function to update the recommendation model
def update_model():
    try:
        # Load behavior data from environment variable or default path
        data_path = os.getenv('BEHAVIOR_DATA_PATH', 'path_to_your_data.csv')
        logging.info(f"Loading behavior data from {data_path}...")
        behavior_data = load_data(data_path)

        # Build interaction matrix
        interaction_matrix = build_interaction_matrix(behavior_data)

        # Train NMF model with configurable components
        n_components = int(os.getenv('NMF_COMPONENTS', 15))  # Default to 15 components
        logging.info(f"Training NMF model with {n_components} components...")
        model, user_features, item_features = train_nmf_model(interaction_matrix, n_components)

        logging.info("Model updated successfully.")
        return model, user_features, item_features
    except Exception as e:
        logging.error(f"Error occurred while updating the model: {e}")
        raise


# Function to validate the recommendation model
def validate_model(**kwargs):
    model, user_features, item_features = kwargs['ti'].xcom_pull(task_ids='update_model_task')
    # Implement validation logic here, e.g., cross-validation or testing with a holdout set
    logging.info("Validating the model...")
    validation_score = model.reconstruction_err_  # Example: Use reconstruction error as a validation metric
    logging.info(f"Model validation score: {validation_score}")


# Function to backup model and interaction matrix to S3
def backup_model_to_s3(**kwargs):
    s3_client = boto3.client('s3')
    model, user_features, item_features = kwargs['ti'].xcom_pull(task_ids='update_model_task')

    try:
        # Convert model components to byte format and upload to S3
        bucket_name = os.getenv('S3_BUCKET_NAME', 'your-bucket-name')
        model_backup_path = 'backups/model_backup.pkl'
        interaction_backup_path = 'backups/interaction_matrix_backup.csv'

        # Example backup of model as pkl
        s3_client.put_object(Bucket=bucket_name, Key=model_backup_path, Body=str(model))
        logging.info(f"Model backed up to S3: {model_backup_path}")

        # Example backup of user-item interaction matrix as CSV
        interaction_matrix = kwargs['ti'].xcom_pull(task_ids='update_model_task')[1]  # interaction_matrix
        interaction_matrix.to_csv(interaction_backup_path)
        s3_client.upload_file(interaction_backup_path, bucket_name, interaction_backup_path)
        logging.info(f"Interaction matrix backed up to S3: {interaction_backup_path}")
    except Exception as e:
        logging.error(f"Error occurred while backing up to S3: {e}")
        raise


# Function to clean up old models
def cleanup_old_models():
    logging.info("Cleaning up old models...")
    # Implement logic to remove outdated models or archives from storage


# Dummy start task
start_task = DummyOperator(
    task_id='start_task',
    dag=dag
)

# Task to update the recommendation model
update_model_task = PythonOperator(
    task_id='update_model_task',
    python_callable=update_model,
    dag=dag,
    retries=3,  # Retry up to 3 times if task fails
)

# Task to validate the updated model
validate_model_task = PythonOperator(
    task_id='validate_model_task',
    python_callable=validate_model,
    provide_context=True,
    dag=dag,
)

# Task to back up the model and interaction matrix to S3
backup_model_task = PythonOperator(
    task_id='backup_model_task',
    python_callable=backup_model_to_s3,
    provide_context=True,
    dag=dag,
)

# Task to clean up old models
cleanup_old_models_task = PythonOperator(
    task_id='cleanup_old_models_task',
    python_callable=cleanup_old_models,
    dag=dag,
)

# Task to send a notification upon successful model update
notify_success_task = EmailOperator(
    task_id='notify_success_task',
    to='youremail@example.com',
    subject='Model Update Successful',
    html_content='<p>The recommendation model has been successfully updated and validated.</p>',
    dag=dag
)

# Task dependencies
start_task >> update_model_task >> validate_model_task >> backup_model_task >> cleanup_old_models_task >> notify_success_task
