import logging
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from self_utils import read_csi_data_from_csv, read_labels_from_csv
from loader import CSIDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training_models.log"), logging.StreamHandler()],
)

DATASET_FOLDER = "./preprocessing/merged"
SEQ_DIM = 1024
DATA_STEP = 8
MODEL_PATH = "saved_models/best_model.pkl"

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

    # Feature Scaling (Normalization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    logging.info(f"Data loaded. Train size: {X_train.shape}, Validation size: {X_val.shape}")
    return X_train, y_train, X_val, y_val

def train():
    """Trains multiple models and selects the best one."""
    logging.info("Loading dataset...")
    X_train, y_train, X_val, y_val = load_data()

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=2, min_samples_leaf=4, bootstrap=False, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
    }

    best_model = None
    best_accuracy = 0.0

    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train, y_train)

        logging.info(f"Evaluating {name}...")
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        logging.info(f"{name} Validation Accuracy: {accuracy * 100:.2f}%")
        print(f"{name} Validation Accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    # Save the best model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    logging.info(f"Best Model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy")
    print(f"Best Model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy")

if __name__ == "__main__":
    train()
