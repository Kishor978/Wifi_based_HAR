import logging
import os
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset_loader import CSIDataset
from metrics import get_train_metric
from models import LSTMClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np



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
DATASET_FOLDER = "/kaggle/input/csi-data/dataset"
DATA_ROOMS = ["bedroom_lviv", "parents_home", "vitalnia_lviv"]
DATA_SUBROOMS = [["1", "2", "3", "4"], ["1"], ["1", "2", "3", "4", "5"]]

# LSTM Model parameters
input_dim = 468  # 114 subcarriers * 4 antenna_pairs * 2 (amplitude + phase)
hidden_dim = 256
layer_dim = 2
output_dim = 5
dropout_rate = 0.5
bidirectional = False
SEQ_DIM = 1024
DATA_STEP = 8

BATCH_SIZE = 16
EPOCHS_NUM = 100
LEARNING_RATE = 0.00146

class_weights = (
    torch.Tensor([0.113, 0.439, 0.1515, 0.1212, 0.1363])
    .double()
    .to(device)
)
class_weights_inv = 1 / class_weights
logging.info("class_weights_inv: {}".format(class_weights_inv))


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


def get_session_majority_class(session_path):
    """Load labels from a session and return its majority class."""
    # Load labels for this session (modify based on your data structure)
    # Example: Assuming CSIDataset can load a single session
    dataset = CSIDataset([session_path], window_size=1, step=1)
    labels = [dataset.labels[i] for i in range(len(dataset))]
    majority_class = max(set(labels), key=labels.count)
    return majority_class


def load_data():
    # List all sessions
    all_sessions = []
    for room_idx, room in enumerate(DATA_ROOMS):
        for subroom in DATA_SUBROOMS[room_idx]:
            session_path = os.path.join(DATASET_FOLDER, room, subroom)
            all_sessions.append(session_path)

    # Get majority class for each session
    session_labels = []
    for session in all_sessions:
        print("session: ", session)
        majority_class = get_session_majority_class(session)
        session_labels.append(majority_class)

    # Stratified split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(split.split(all_sessions, session_labels))
    train_sessions = [all_sessions[i] for i in train_indices]
    val_sessions = [all_sessions[i] for i in val_indices]

    # Create datasets
    train_dataset = CSIDataset(
        train_sessions, window_size=SEQ_DIM, step=DATA_STEP, is_training=True
    )
    val_dataset = CSIDataset(
        val_sessions, window_size=SEQ_DIM, step=DATA_STEP, is_training=False
    )

    # Normalize using training stats
    train_mean = np.mean(train_dataset.amplitudes)
    train_std = np.std(train_dataset.amplitudes)
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
        BATCH_SIZE,
    )
    model = model.double().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_inv)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )  # L2 regularization
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.5,
        patience=5,
    )

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

    # training loop
    logging.info("Start model training")
    for epoch in range(1, EPOCHS_NUM + 1):
        model.train(mode=True)
        correct_predictions = 0
        total_predictions = 0
        epoch_loss = 0
        # model.hidden = model.init_hidden(BATCH_SIZE)
        for i, (x_batch, y_batch) in tqdm(
            enumerate(trn_dl), total=len(trn_dl), desc="Training epoch: "
        ):
            if x_batch.size(0) != BATCH_SIZE:
                logging.warning(
                    f"Skipping batch {i} due to inconsistent size: {x_batch.size(0)}"
                )

                continue
            print("Training epoch: ", epoch)
            model.init_hidden(x_batch.size(0))
            x_batch, y_batch = x_batch.double().to(device), y_batch.double().to(device)

            # Forward pass
            out = model(x_batch)

            loss = criterion(out, y_batch.long())
            epoch_loss += loss.item()

            # zero the parameter gradients
            optimizer.zero_grad()

            # Backward and optimize
            loss.backward()
            optimizer.step()
            # Metrics
            _, predicted = torch.max(out, 1)
            correct_predictions += (predicted == y_batch).sum().item()
            total_predictions += y_batch.size(0)
            if i % 50 == 0:  # Debugging step every 10 batches
                logging.info(f"Epoch {epoch}, Batch {i}: Loss = {loss.item():.4f}")

        train_accuracy = correct_predictions / total_predictions

        val_loss, val_correct, val_total, val_acc = get_train_metric(
            model, val_dl, criterion, BATCH_SIZE
        )

        logging.info(
            f"Epoch: {epoch:3d} |"
            f" Validation Loss: {val_loss/len(val_dl):.4f}, Validation Acc.: {val_acc:2.2%}, "
            f"Train Loss: {epoch_loss / len(trn_dl):.4f}, Train Acc: {train_accuracy:.2%}"
        )

        print(            f"Epoch: {epoch:3d} |"
            f" Validation Loss: {val_loss/len(val_dl):.4f}, Validation Acc.: {val_acc:2.2%}, "
            f"Train Loss: {epoch_loss / len(trn_dl):.4f}, Train Acc: {train_accuracy:.2%}"
)
        if val_acc > best_acc:
            trials = 0
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "lstm_classifier_best.pth"))
            logging.info(
                f"Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}"
            )
        else:
            trials += 1
            if trials >= patience:
                logging.info(f"Early stopping on epoch {epoch}")
                break

        # Save checkpoint after each epoch
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
