import logging
import os
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from dataset_loader import CSIDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("prediction.log"), logging.StreamHandler()],
)

MODEL_PATH = "saved_models/best_random_forest.pkl"
DATASET_FOLDER = "./dataset"
SEQ_DIM = 1024
DATA_STEP = 8
CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]


def load_model():
    """Load the trained Random Forest model"""
    if not os.path.exists(MODEL_PATH):
        logging.error("Trained model not found! Please train the model first.")
        return None
    logging.info(f"Loading model from {MODEL_PATH}...")
    return joblib.load(MODEL_PATH)


def extract_features_and_labels(sessions):
    """Extract features (flattened CSI data) and labels from given sessions"""
    dataset = CSIDataset(sessions, window_size=SEQ_DIM, step=DATA_STEP, is_training=False)
    features = []
    labels = []

    for x_batch, y_batch in dataset:
        features.append(x_batch.flatten())  # Convert tensor to NumPy array
        labels.append(y_batch)  # Convert tensor label to integer

    return np.array(features), np.array(labels)


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix using seaborn"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()


def predict(sessions):
    """Make predictions using the trained Random Forest model and evaluate performance"""
    model = load_model()
    if model is None:
        return

    logging.info("Extracting features for prediction...")
    X_test, y_true = extract_features_and_labels(sessions)

    logging.info(f"Making predictions on {X_test.shape[0]} samples...")
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")

    # Display confusion matrix
    plot_confusion_matrix(y_true, y_pred)

    # Output individual predictions
    for i, pred in enumerate(y_pred):
        logging.info(f"Sample {i + 1}: Predicted -> {CLASS_NAMES[pred]}, Actual -> {CLASS_NAMES[y_true[i]]}")

    return y_pred


if __name__ == "__main__":
    # Example test session paths (adjust based on dataset structure)
    test_sessions = [os.path.join(DATASET_FOLDER, "bedroom_lviv", "1")]
    predict(test_sessions)
