import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    :param file_path: path to the csv file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path} with shape {df.shape}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the CSV file: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured while loading data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train the RandomForestClassifier

    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"The number of samples in labels and features must be same")
        
        logger.debug(f"Initializing RandomForestClassifier model with parameters: {params}")
        clf = RandomForestClassifier(n_estimators = params["n_estimators"], random_state = params["random_state"])
        logger.debug(f"Model traning started with {X_train.shape[0]} samples.")

        clf.fit(X_train, y_train)
        logger.debug(f"Model training completed")

        return clf
    except ValueError as e:
        logger.error(f"ValueError during model training: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during trainig model")
        raise

def save_model(model: RandomForestClassifier, file_path: str) -> None:
    """
    Save the model to the trained file.

    :param model: Trained model object
    :param file_path: Path to save the model file
    """

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        
        logger.debug(f"Model saved to {file_path}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error("Unexpected error in saving the mode")
        raise

def main():
    try:
        params = {"n_estimators": 25, "random_state": 2}
        train_data = load_data("./data/processed/train_tfidf.csv")
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)

        model_save_path = "models/model.pickle"
        save_model(clf, model_save_path)
    except Exception as e:
        logger.error(f"Failed to complete model building process: {e}")
        print(f"Error : {e}")

if __name__ == '__main__':
    main()