import os
import boto3
import logging
from botocore.exceptions import NoCredentialsError, ClientError

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Configure AWS CloudWatch Client
def get_cloudwatch_client():
    try:
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        cloudwatch = boto3.client('cloudwatch', region_name=aws_region)
        logging.info(f"Initialized CloudWatch client in {aws_region} region.")
        return cloudwatch
    except NoCredentialsError:
        logging.error("AWS credentials not found. Please configure your AWS credentials.")
        raise
    except Exception as e:
        logging.error(f"Error initializing CloudWatch client: {e}")
        raise


# Log metric to CloudWatch
def log_metric(metric_name, value, unit='Count'):
    try:
        cloudwatch = get_cloudwatch_client()
        namespace = os.getenv('CLOUDWATCH_NAMESPACE', 'RecommendationSystem')

        response = cloudwatch.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Value': value,
                    'Unit': unit
                }
            ]
        )
        logging.info(f"Metric '{metric_name}' logged with value {value} {unit} to namespace '{namespace}'.")
        return response
    except ClientError as e:
        logging.error(f"Failed to log metric {metric_name}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error logging metric {metric_name}: {e}")
        raise


# Function to log latency metrics
def log_latency(latency):
    try:
        log_metric('ModelPredictionLatency', latency, 'Seconds')
        logging.info(f"Logged latency: {latency} seconds.")
    except Exception as e:
        logging.error(f"Error logging latency: {e}")
        raise


# Example function to log a custom metric (e.g., model accuracy)
def log_model_accuracy(accuracy):
    try:
        log_metric('ModelAccuracy', accuracy, 'Percent')
        logging.info(f"Logged model accuracy: {accuracy}%.")
    except Exception as e:
        logging.error(f"Error logging model accuracy: {e}")
        raise


if __name__ == "__main__":
    # Example of logging model prediction latency
    latency = 0.123  # Example latency in seconds
    log_latency(latency)

    # Example of logging model accuracy
    accuracy = 95.67  # Example accuracy in percentage
    log_model_accuracy(accuracy)
