import os
import json
import logging
from recommendation_model import get_recommendations, load_data, train_nmf_model

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to load data and train the model (lazy loading)
def load_model():
    try:
        # Load data from the environment variable or use default path
        data_path = os.getenv('BEHAVIOR_DATA_PATH', 'path_to_your_data.csv')
        logging.info(f"Loading behavior data from {data_path}...")
        interaction_matrix = load_data(data_path)

        # Train the NMF model with configurable number of components
        n_components = int(os.getenv('NMF_COMPONENTS', 15))  # Default to 15 components
        logging.info(f"Training NMF model with {n_components} components...")
        model, user_features, item_features = train_nmf_model(interaction_matrix, n_components)

        logging.info("Model loaded and trained successfully.")
        return model, user_features, item_features
    except Exception as e:
        logging.error(f"Error loading data or training model: {e}")
        raise


# Lambda handler
def lambda_handler(event, context):
    try:
        # Validate input
        if 'user_id' not in event:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing user_id in the request.'})
            }

        user_id = event['user_id']
        if not isinstance(user_id, int) or user_id < 0:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid user_id. It must be a non-negative integer.'})
            }

        # Load the model and data dynamically (if not already loaded)
        model, user_features, item_features = load_model()

        # Get recommendations
        recommendations = get_recommendations(user_features, item_features, user_id)

        # Return recommendations as JSON response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'recommendations': recommendations.tolist()
            })
        }

    except IndexError:
        logging.error(f"User ID {user_id} is out of bounds.")
        return {
            'statusCode': 404,
            'body': json.dumps({'error': f'User ID {user_id} is out of bounds.'})
        }
    except Exception as e:
        logging.error(f"Error during recommendation process: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'An internal error occurred while processing your request.'})
        }

