import os
import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import NMF
from sklearn.exceptions import NotFittedError

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Load data function
def load_data(path):
    try:
        df = pd.read_csv(path)
        logging.info(f"Data loaded successfully from {path}.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {path}. Please check the path.")
        raise
    except pd.errors.EmptyDataError:
        logging.error("Loaded data is empty.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        raise


# Build interaction matrix
def build_interaction_matrix(df):
    if df.empty or 'user_id' not in df.columns or 'item_id' not in df.columns or 'interaction' not in df.columns:
        logging.error("Invalid data format. Dataframe must contain 'user_id', 'item_id', and 'interaction' columns.")
        raise ValueError("Invalid data format.")
    logging.info("Building user-item interaction matrix...")
    interaction_matrix = df.pivot_table(index='user_id', columns='item_id', values='interaction').fillna(0)
    logging.info("Interaction matrix successfully built.")
    return interaction_matrix


# Train NMF model
def train_nmf_model(interaction_matrix, n_components=15):
    logging.info(f"Training NMF model with {n_components} components...")
    try:
        model = NMF(n_components=n_components, random_state=42)
        user_features = model.fit_transform(interaction_matrix)
        item_features = model.components_
        logging.info("NMF model training complete.")
        return model, user_features, item_features
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise


# Generate recommendations for a specific user
def get_recommendations(user_features, item_features, user_id):
    try:
        # Ensure the user_id is within the correct range
        if user_id >= user_features.shape[0] or user_id < 0:
            logging.error(f"Invalid user_id: {user_id}. It must be between 0 and {user_features.shape[0] - 1}.")
            raise IndexError(f"User ID {user_id} is out of range.")

        # Compute the recommendation scores
        recommendations = np.dot(user_features, item_features)

        # Sort items by their predicted interaction scores in descending order
        recommended_items = recommendations[user_id].argsort()[::-1]
        logging.info(f"Recommendations for user {user_id} generated successfully.")
        return recommended_items
    except IndexError as e:
        logging.error(f"User ID is out of bounds: {e}")
        raise
    except NotFittedError:
        logging.error("Model not fitted yet. Ensure the model has been trained before generating recommendations.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while generating recommendations: {e}")
        raise


if __name__ == "__main__":
    try:
        # Load behavior data from the provided path
        data_path = os.getenv('BEHAVIOR_DATA_PATH', 'path_to_your_data.csv')
        behavior_data = load_data(data_path)

        # Build the user-item interaction matrix
        interaction_matrix = build_interaction_matrix(behavior_data)

        # Train NMF model with configurable number of components
        n_components = int(os.getenv('NMF_COMPONENTS', 15))  # Default to 15 components
        model, user_features, item_features = train_nmf_model(interaction_matrix, n_components)

        # Generate recommendations for a specific user (ID provided via environment variable or default to 0)
        user_id = int(os.getenv('USER_ID', 0))  # Default user ID to 0
        recommended_items = get_recommendations(user_features, item_features, user_id)

        logging.info(f"Top recommended items for user {user_id}: {recommended_items}")

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")

