import os
import logging
from flask import Flask, request, jsonify
from recommendation_model import get_recommendations, load_data, train_nmf_model

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Global variables for user-item matrix and model components
interaction_matrix = None
user_features = None
item_features = None
model = None


# Load data and train model (Lazy loading on first request)
def initialize_model():
    global interaction_matrix, user_features, item_features, model
    if interaction_matrix is None or user_features is None or item_features is None:
        try:
            # Load data from environment variable or default path
            data_path = os.getenv('BEHAVIOR_DATA_PATH', 'path_to_your_data.csv')
            logging.info(f"Loading behavior data from {data_path}...")
            interaction_matrix = load_data(data_path)

            # Train NMF model with configurable components
            n_components = int(os.getenv('NMF_COMPONENTS', 15))  # Default to 15 components
            logging.info(f"Training NMF model with {n_components} components...")
            model, user_features, item_features = train_nmf_model(interaction_matrix, n_components)
            logging.info("Model training complete.")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise


# Route to get recommendations for a user
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Initialize the model if not already loaded
        initialize_model()

        # Validate incoming request
        if not request.is_json or 'user_id' not in request.json:
            return jsonify({'error': 'Invalid request format. Must provide user_id in JSON format.'}), 400

        user_id = request.json['user_id']
        if not isinstance(user_id, int) or user_id < 0:
            return jsonify({'error': 'Invalid user_id. It must be a non-negative integer.'}), 400

        # Get recommendations
        recommendations = get_recommendations(user_features, item_features, user_id)
        return jsonify({'recommendations': recommendations.tolist()}), 200

    except IndexError:
        logging.error(f"User ID {user_id} is out of bounds.")
        return jsonify({'error': f'User ID {user_id} is out of bounds.'}), 404
    except Exception as e:
        logging.error(f"An error occurred while generating recommendations: {e}")
        return jsonify({'error': 'An internal error occurred while processing your request.'}), 500


if __name__ == '__main__':
    # Get host and port from environment variables
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')  # Default to running on all interfaces
    port = int(os.getenv('FLASK_RUN_PORT', 5000))  # Default to port 5000

    logging.info(f"Starting Flask server on {host}:{port}...")
    app.run(debug=os.getenv('FLASK_DEBUG', 'True') == 'True', host=host, port=port)
