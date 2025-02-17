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
from xgboost import XGBClassifier
import pickle


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


def extract_features_and_labels(dataset, batch_size=512):
    """Generator function to yield data in batches."""
    num_samples = len(dataset)
    
    for i in range(0, num_samples, batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]

        X_batch, y_batch = zip(*batch)  # Unpack batch
        
        X_batch = np.array(X_batch)  # Convert list to NumPy array
        y_batch = np.array(y_batch)  # Convert labels to NumPy array
        
        # Ensure X_batch is 2D (batch_size, num_features)
        X_batch = X_batch.reshape(X_batch.shape[0], -1)

        yield X_batch, y_batch  # Yield batch properly
        
        
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
    """Loads the dataset and processes it in batches to avoid memory overflow."""
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

    logging.info("Extracting training features in batches...")
    train_batches = extract_features_and_labels(train_dataset, batch_size=512)
    X_train, y_train = zip(*train_batches)  # Convert generator output into lists

    logging.info("Extracting validation features in batches...")
    val_batches = extract_features_and_labels(val_dataset, batch_size=512)
    X_val, y_val = zip(*val_batches)

    # Convert lists of batches to numpy arrays
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    X_val = np.vstack(X_val)
    y_val = np.hstack(y_val)

    logging.info(f"Data loaded. Train size: {X_train.shape}, Validation size: {X_val.shape}")
    return X_train, y_train, X_val, y_val
    # model = RandomForestClassifier(n_estimators=N_ESTIMATORS,max_depth=MAX_DEPTH,min_samples_leaf=MIN_SAMPLES_LEAF,min_samples_split=MIN_SAMPLES_SPLIT,bootstrap=BOOTSTRAP,random_state=42, n_jobs=-1)

def train():
    """Trains the Random Forest model using batch-wise processing to avoid memory overflow."""
    logging.info("Loading dataset...")
    
    # Load datasets in batches
    X_train_batches, y_train_batches, X_val_batches, y_val_batches = load_data()

    logging.info("Initializing Random Forest...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
        bootstrap=BOOTSTRAP,
        random_state=42,
        n_jobs=-1
    )

    logging.info("Training model in batches...")
    for X_batch, y_batch in zip(X_train_batches, y_train_batches):
        model.fit(X_batch, y_batch)  # Fit model batch-wise to avoid memory issues
    
    logging.info("Evaluating on validation set...")
    y_preds = []
    
    for X_batch in X_val_batches:
        y_preds.append(model.predict(X_batch))  # Predict batch-wise
    
    y_preds = np.hstack(y_preds)  # Combine batch-wise predictions

    accuracy = accuracy_score(np.hstack(y_val_batches), y_preds)  # Calculate validation accuracy
    logging.info(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    save_path = "saved_models/random_forest_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    
    logging.info(f"Model saved to {save_path}")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()