# import logging
# import os
# import numpy as np
# import joblib
# import torch
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score
# from dataset_loader import CSIDataset

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("training_grid_search.log"), logging.StreamHandler()],
# )

# DATASET_FOLDER = "./dataset"
# DATA_ROOMS = ["bedroom_lviv", "parents_home", "vitalnia_lviv"]
# DATA_SUBROOMS = [["1", "2", "3", "4"], ["1"], ["1", "2", "3", "4", "5"]]

# SEQ_DIM = 1024
# DATA_STEP = 8
# MODEL_PATH = "saved_models/best_random_forest.pkl"


# def extract_features_and_labels(sessions, is_training=True):
#     """Extract features (flattened CSI data) and labels"""
#     dataset = CSIDataset(sessions, window_size=SEQ_DIM, step=DATA_STEP, is_training=is_training)
#     features, labels = [], []

#     for x_batch, y_batch in dataset:
#         features.append(x_batch.flatten())  # Convert tensor to NumPy array
#         labels.append(y_batch) # Convert tensor to scalar

#     return np.array(features), np.array(labels)


# def load_data():
#     """Loads the dataset and splits it into training and validation sets"""
#     all_sessions = [
#         os.path.join(DATASET_FOLDER, room, subroom)
#         for room_idx, room in enumerate(DATA_ROOMS)
#         for subroom in DATA_SUBROOMS[room_idx]
#     ]

#     train_sessions, val_sessions = train_test_split(all_sessions, test_size=0.2, random_state=42, shuffle=True)

#     logging.info("Extracting training features...")
#     X_train, y_train = extract_features_and_labels(train_sessions, is_training=True)

#     logging.info("Extracting validation features...")
#     X_val, y_val = extract_features_and_labels(val_sessions, is_training=False)

#     logging.info(f"Data loaded. Train size: {X_train.shape}, Validation size: {X_val.shape}")
#     return X_train, y_train, X_val, y_val


# def train():
#     """Performs hyperparameter tuning using Grid Search and trains the best model"""
#     X_train, y_train, X_val, y_val = load_data()

#     logging.info("Initializing Random Forest...")
#     rf = RandomForestClassifier(random_state=42, n_jobs=-1)

#     # Define hyperparameter grid
#     param_grid = {
#         "n_estimators": [50, 100, 200],  # Number of trees
#         "max_depth": [None, 10, 20],  # Maximum depth of trees
#         "min_samples_split": [2, 5, 10],  # Minimum samples to split a node
#         "min_samples_leaf": [1, 2, 4],  # Minimum samples in leaf node
#         "bootstrap": [True, False],  # Whether to use bootstrap sampling
#     }

#     # Grid Search with 5-fold cross-validation
#     grid_search = GridSearchCV(
#         estimator=rf,
#         param_grid=param_grid,
#         scoring="accuracy",
#         cv=5,
#         verbose=2,
#         n_jobs=-1
#     )

#     logging.info("Performing Grid Search CV...")
#     grid_search.fit(X_train, y_train)

#     # Get the best model and parameters
#     best_model = grid_search.best_estimator_
#     best_params = grid_search.best_params_
#     logging.info(f"Best Parameters: {best_params}")

#     # Evaluate on validation data
#     y_pred = best_model.predict(X_val)
#     val_acc = accuracy_score(y_val, y_pred)
#     logging.info(f"Validation Accuracy with Best Model: {val_acc:.2%}")

#     # Save best model
#     os.makedirs("saved_models", exist_ok=True)
#     joblib.dump(best_model, MODEL_PATH)
#     logging.info(f"Best model saved to {MODEL_PATH}")


# if __name__ == "__main__":
#     train()

import logging
import os
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from dataset_loader import CSIDataset
import joblib  # For saving the model
from utils import  read_csi_data_from_csv, read_labels_from_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training_random_forest.log"), logging.StreamHandler()],
)

DATASET_FOLDER = "./notebooks"
SEQ_DIM = 1024
DATA_STEP = 8
N_ESTIMATORS = 100  # Number of trees in Random Forest
BOOTSTRAP = False
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 4
MAX_DEPTH = 20
BATCH_SIZE = 16  # Not used directly, but for feature extraction
MODEL_PATH = "saved_models/random_forest_model.pkl"


def extract_features_and_labels(dataset):
    """Extract features (flattened CSI data) and labels from dataset"""
    features, labels = [], []

    for x_batch, y_batch in dataset:
        features.append(x_batch.flatten())  # Convert tensor to NumPy array
        labels.append(y_batch)  # Convert tensor to scalar
    
    return np.array(features), np.array(labels)
def read_all_data_from_files(data_path, label_path, is_five_hhz=True, antenna_pairs=4):
    """
    Read CSI and labels from merged CSV files.
    """
    amplitudes, phases = read_csi_data_from_csv(data_path, is_five_hhz, antenna_pairs)
    labels, valid_indices = read_labels_from_csv(label_path, len(amplitudes))
    # # print(len(valid_indices))
    # # print(len(amplitudes))
    # print(phases.shape)
    # Apply the filter
    amplitudes, phases = amplitudes[valid_indices], phases[valid_indices]

    return amplitudes, phases, labels


def load_data():
    """Loads the dataset and splits it into training and validation sets"""
    # Load merged CSI data and labels
    data_path = os.path.join(DATASET_FOLDER, "data.csv")
    label_path = os.path.join(DATASET_FOLDER, "label.csv")
    amplitudes, phases, labels = read_all_data_from_files(data_path, label_path)
    # Concatenate amplitude and phase data for input features
    csi_data = np.hstack((amplitudes, phases))

    # Stratified split
    train_csi, val_csi, train_labels, val_labels = train_test_split(
        csi_data, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Create datasets
    train_dataset = CSIDataset(
        train_csi, train_labels, window_size=SEQ_DIM, step=DATA_STEP, is_training=True
    )
    val_dataset = CSIDataset(
        val_csi, val_labels, window_size=SEQ_DIM, step=DATA_STEP, is_training=False
    )
    logging.info("Extracting training features...")
    X_train, y_train = extract_features_and_labels(train_dataset)

    logging.info("Extracting validation features...")
    X_val, y_val = extract_features_and_labels(val_dataset)

    logging.info(f"Data loaded. Train size: {X_train.shape}, Validation size: {X_val.shape}")
    return X_train, y_train, X_val, y_val


def train():
    """Trains a Random Forest classifier"""
    X_train, y_train, X_val, y_val = load_data()

    logging.info("Initializing Random Forest...")
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS,max_depth=MAX_DEPTH,min_samples_leaf=MIN_SAMPLES_LEAF,min_samples_split=MIN_SAMPLES_SPLIT,bootstrap=BOOTSTRAP,random_state=42, n_jobs=-1)

    logging.info("Training model...")
    model.fit(X_train, y_train)

    # Evaluate on validation data
    y_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)

    logging.info(f"Validation Accuracy: {val_acc:.2%}")
    print(f"Validation Accuracy: {val_acc:.2%}")
    # Save model
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()