import logging
import os
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from dataset_loader import CSIDataset
import joblib  # For saving the model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training_random_forest.log"), logging.StreamHandler()],
)

DATASET_FOLDER = "./dataset"
DATA_ROOMS = ["bedroom_lviv", "parents_home", "vitalnia_lviv"]
DATA_SUBROOMS = [["1", "2", "3", "4"], ["1"], ["1", "2", "3", "4", "5"]]

SEQ_DIM = 1024
DATA_STEP = 8
N_ESTIMATORS = 100  # Number of trees in Random Forest
BATCH_SIZE = 16  # Not used directly, but for feature extraction
MODEL_PATH = "saved_models/random_forest_model.pkl"


def extract_features_and_labels(sessions, is_training=True):
    """Extract features (flattened CSI data) and labels from dataset"""
    dataset = CSIDataset(sessions, window_size=SEQ_DIM, step=DATA_STEP, is_training=is_training)
    features, labels = [], []

    for x_batch, y_batch in dataset:
        features.append(x_batch.flatten())  # Convert tensor to NumPy array
        labels.append(y_batch)  # Convert tensor to scalar
    
    return np.array(features), np.array(labels)


def load_data():
    """Loads the dataset and splits it into training and validation sets"""
    all_sessions = [
        os.path.join(DATASET_FOLDER, room, subroom)
        for room_idx, room in enumerate(DATA_ROOMS)
        for subroom in DATA_SUBROOMS[room_idx]
    ]

    train_sessions, val_sessions = train_test_split(all_sessions, test_size=0.2, random_state=42, shuffle=True)

    logging.info("Extracting training features...")
    X_train, y_train = extract_features_and_labels(train_sessions, is_training=True)

    logging.info("Extracting validation features...")
    X_val, y_val = extract_features_and_labels(val_sessions, is_training=False)

    logging.info(f"Data loaded. Train size: {X_train.shape}, Validation size: {X_val.shape}")
    return X_train, y_train, X_val, y_val


def train():
    """Trains a Random Forest classifier"""
    X_train, y_train, X_val, y_val = load_data()

    logging.info("Initializing Random Forest...")
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)

    logging.info("Training model...")
    model.fit(X_train, y_train)

    # Evaluate on validation data
    y_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)

    logging.info(f"Validation Accuracy: {val_acc:.2%}")

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
