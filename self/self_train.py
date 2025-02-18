import logging
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from loader import CSIDataset
from metrics import get_train_metric
from LSTM_classifier import LSTMClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from self_utils import read_csi_data_from_csv, read_labels_from_csv


# Configure logging
log_filename = "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Log to a file
        logging.StreamHandler(),  # Log to the console
    ],
)

# Cuda support
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

logging.info("Device: {}".format(device))

# Define dataset structure
DATASET_FOLDER = "/kaggle/input/my-csi"

# LSTM Model parameters
input_dim = 64  
hidden_dim = 256
layer_dim = 2
output_dim = 4
dropout_rate = 0.2
bidirectional = False
SEQ_DIM = 1024
DATA_STEP = 8

BATCH_SIZE = 16
EPOCHS_NUM = 100
LEARNING_RATE = 0.001

# class_weights = torch.Tensor([0.4225, 1.3319, 1.4432, 1.6811, 1.6820]).double().to(device)
# class_weights_inv = 1 / class_weights
# logging.info("class_weights_inv: {}".format(class_weights_inv))


def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    logging.info(f"Checkpoint saved: {filename}")


def load_checkpoint(filename="checkpoint.pth"):
    if torch.cuda.is_available():
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=torch.device("cpu"))

    logging.info(f"Checkpoint loaded: {filename}")
    return checkpoint


def read_all_data_from_files(data_path, label_path,  antenna_pairs=1):
    """
    Read CSI and labels from merged CSV files.
    """
    amplitudes, phases = read_csi_data_from_csv(data_path,  antenna_pairs)
    labels = read_labels_from_csv(label_path, len(amplitudes))
    # # print(len(valid_indices))
    # # print(len(amplitudes))
    # print(phases.shape)
    # Apply the filter
    amplitudes, phases = amplitudes[:], phases[:]

    return amplitudes, phases, labels


def load_data():
    # Load merged CSI data and labels
    data_path = os.path.join(DATASET_FOLDER, "data.csv")
    label_path = os.path.join(DATASET_FOLDER, "label.csv")
    amplitudes, phases, labels = read_all_data_from_files(data_path, label_path)
    print("label shape",labels.shape)
    print("amplitudes shape",amplitudes.shape)
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

    # Normalize using training stats
    train_mean = np.mean(train_csi)
    train_std = np.std(train_csi)
    train_dataset.normalize_mean = train_mean
    train_dataset.normalize_std = train_std
    val_dataset.normalize_mean = train_mean
    val_dataset.normalize_std = train_std

    # DataLoaders
    trn_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )  # No shuffle for val!

    return trn_dl, val_dl


def train():
    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    patience, trials, best_acc = 100, 0, 0
    trn_dl, val_dl = load_data()

    start_epoch = 1
    model = LSTMClassifier(
        input_dim,
        hidden_dim,
        layer_dim,
        dropout_rate,
        bidirectional,
        output_dim,
    )
    model = model.double().to(device)

    criterion = nn.CrossEntropyLoss()  # Apply class weights if needed
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    # Load checkpoint if available
    checkpoint_filename = "checkpoint.pth"
    try:
        checkpoint = load_checkpoint(checkpoint_filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        logging.info(f"Resuming from epoch {start_epoch}")

    except FileNotFoundError:
        start_epoch = 1
        logging.info("No checkpoint found, starting training from scratch.")

    # Training loop
    logging.info("Start model training")
    for epoch in range(start_epoch, EPOCHS_NUM + 1):
        model.train()
        correct_predictions, total_predictions = 0, 0
        epoch_loss = 0

        for i, (x_batch, y_batch) in tqdm(enumerate(trn_dl), total=len(trn_dl), desc=f"Training Epoch {epoch}"):
            if x_batch.size(0) != BATCH_SIZE:
                logging.warning(f"Skipping batch {i} due to inconsistent size: {x_batch.size(0)}")
                continue

            x_batch, y_batch = x_batch.double().to(device), y_batch.long().to(device)  # Ensure correct data types

            # Forward pass
            out = model(x_batch)
            loss = criterion(out, y_batch)
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(out, 1)
            correct_predictions += (predicted == y_batch).sum().item()
            total_predictions += y_batch.size(0)

            if i % 50 == 0:
                logging.info(f"Epoch {epoch}, Batch {i}: Loss = {loss.item():.4f}")

        train_accuracy = correct_predictions / total_predictions

        # Validate
        val_loss, val_correct, val_total, val_acc = get_train_metric(model, val_dl, criterion, BATCH_SIZE)

        logging.info(
            f"Epoch: {epoch:3d} |"
            f" Train Loss: {epoch_loss / len(trn_dl):.4f}, Train Acc: {train_accuracy:.2%} | "
            f"Validation Loss: {val_loss/len(val_dl):.4f}, Validation Acc.: {val_acc:2.2%}"
        )

        print(
            f"Epoch: {epoch:3d} |"
            f" Train Loss: {epoch_loss / len(trn_dl):.4f}, Train Acc: {train_accuracy:.2%} | "
            f"Validation Loss: {val_loss/len(val_dl):.4f}, Validation Acc.: {val_acc:2.2%}"
        )

        # Save best model
        if val_acc > best_acc:
            trials = 0
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "lstm_classifier_best.pth"))
            logging.info(f"Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}")
        else:
            trials += 1
            if trials >= patience:
                logging.info(f"Early stopping on epoch {epoch}")
                break

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
            },
            checkpoint_filename,
        )

        scheduler.step(val_loss)

if __name__ == "__main__":
    train()