import logging
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from self_utils import read_csi_data_from_csv, read_labels_from_csv
from loader import CSIDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training_random_forest.log"), logging.StreamHandler()],
)

DATASET_FOLDER = "./preprocessing/merged"
SEQ_DIM = 1024
DATA_STEP = 8
N_ESTIMATORS = 100  
BOOTSTRAP = False
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 4
MAX_DEPTH = 20
MODEL_PATH = "saved_models/random_forest_model.pkl"

def extract_features_and_labels(dataset, batch_size=512):
    """Extracts features from CSI dataset in batches."""
    X_list, y_list = [], []
    
    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        X_batch, y_batch = zip(*batch)
        X_list.append(np.array(X_batch).reshape(len(X_batch), -1))
        y_list.append(np.array(y_batch))
    
    return np.vstack(X_list), np.hstack(y_list)

def read_all_data():
    """Loads CSI and label data from CSV files."""
    data_path = os.path.join(DATASET_FOLDER, "data.csv")
    label_path = os.path.join(DATASET_FOLDER, "label.csv")
    
    amplitudes, phases = read_csi_data_from_csv(data_path, antenna_pairs=1)
    labels = read_labels_from_csv(label_path, len(amplitudes))

    csi_data = np.hstack((amplitudes, phases))
    return train_test_split(csi_data, labels, test_size=0.2, stratify=labels, random_state=42)

def load_data():
    """Prepares CSI dataset with extracted features."""
    train_csi, val_csi, train_labels, val_labels = read_all_data()

    train_dataset = CSIDataset(train_csi, train_labels, SEQ_DIM, DATA_STEP, is_training=True)
    val_dataset = CSIDataset(val_csi, val_labels, SEQ_DIM, DATA_STEP, is_training=False)

    logging.info("Extracting training features...")
    X_train, y_train = extract_features_and_labels(train_dataset)
    
    logging.info("Extracting validation features...")
    X_val, y_val = extract_features_and_labels(val_dataset)

    logging.info(f"Data loaded. Train size: {X_train.shape}, Validation size: {X_val.shape}")
    return X_train, y_train, X_val, y_val

def train():
    """Trains the Random Forest model."""
    logging.info("Loading dataset...")
    X_train, y_train, X_val, y_val = load_data()

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

    logging.info("Training the model...")
    model.fit(X_train, y_train)  # Train on the entire dataset

    logging.info("Evaluating model...")
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    logging.info(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    logging.info(f"Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train()
