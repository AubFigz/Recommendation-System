import os
import pandas as pd
import numpy as np
import logging
from azure.storage.blob import BlobServiceClient, ResourceExistsError
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for large file handling
MAX_FILE_SIZE_MB = 5  # Set file size limit to 5MB per file for uploading


# Function to simulate user behavior data
def simulate_behavior_data(num_users=1000, num_items=500, num_interactions=10000):
    logging.info("Simulating user behavior data...")
    users = np.random.choice(range(num_users), num_interactions)
    items = np.random.choice(range(num_items), num_interactions)
    interactions = np.random.choice([1, 0], num_interactions, p=[0.05, 0.95])
    data = pd.DataFrame({'user_id': users, 'item_id': items, 'interaction': interactions})
    logging.info("Behavior data simulation complete.")
    return data


# Function to anonymize user data by hashing user IDs
def anonymize_data(df):
    logging.info("Anonymizing user data...")
    df['user_id'] = df['user_id'].apply(lambda x: hash(x))
    logging.info("Data anonymization complete.")
    return df


# Function to split large data into chunks for easier upload
def split_dataframe(df, max_file_size_mb):
    logging.info("Checking if data splitting is necessary...")
    csv_size_mb = df.memory_usage(index=True).sum() / (1024 * 1024)
    if csv_size_mb > max_file_size_mb:
        logging.info(f"Data size {csv_size_mb:.2f}MB exceeds {max_file_size_mb}MB, splitting into chunks.")
        num_chunks = int(np.ceil(csv_size_mb / max_file_size_mb))
        return np.array_split(df, num_chunks)
    return [df]


# Function to upload data to Azure Blob Storage
def upload_to_azure(blob_service_client, data, container_name, blob_name):
    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(data.to_csv(index=False), overwrite=True)
        logging.info(f"Data uploaded to Azure Blob Storage as {blob_name}")
    except ResourceExistsError as e:
        logging.error(f"Azure Blob {blob_name} already exists: {e}")
    except Exception as e:
        logging.error(f"Error uploading to Azure Blob Storage: {e}")


# Function to upload data to AWS S3
def upload_to_s3(s3_client, data, bucket_name, file_name):
    try:
        s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=data.to_csv(index=False))
        logging.info(f"Data uploaded to S3 as {file_name}")
    except NoCredentialsError:
        logging.error("AWS credentials not found.")
    except PartialCredentialsError:
        logging.error("Incomplete AWS credentials provided.")
    except Exception as e:
        logging.error(f"Error uploading to S3: {e}")


if __name__ == "__main__":
    # Simulate behavior data and anonymize it
    behavior_data = simulate_behavior_data(num_users=2000, num_items=1000, num_interactions=20000)
    anonymized_data = anonymize_data(behavior_data)

    # Azure Blob Storage upload
    try:
        # Get the Azure Blob Storage connection string from environment variables
        azure_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)

        # Split data into smaller files if necessary
        data_chunks = split_dataframe(anonymized_data, MAX_FILE_SIZE_MB)

        # Upload each chunk as a separate blob (if needed)
        for i, chunk in enumerate(data_chunks):
            blob_name = f'behavior_data_part_{i + 1}.csv' if len(data_chunks) > 1 else 'behavior_data.csv'
            upload_to_azure(blob_service_client, chunk, 'your-container', blob_name)
    except Exception as e:
        logging.error(f"Error uploading to Azure: {e}")

    # AWS S3 upload
    try:
        # Get AWS credentials and set up client
        s3_client = boto3.client('s3')

        # Split data into smaller files if necessary
        data_chunks = split_dataframe(anonymized_data, MAX_FILE_SIZE_MB)

        # Upload each chunk as a separate file in S3 (if needed)
        for i, chunk in enumerate(data_chunks):
            file_name = f'behavior_data_part_{i + 1}.csv' if len(data_chunks) > 1 else 'behavior_data.csv'
            upload_to_s3(s3_client, chunk, 'your-bucket', file_name)
    except Exception as e:
        logging.error(f"Error uploading to S3: {e}")
