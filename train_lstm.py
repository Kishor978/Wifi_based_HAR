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

# Define dataset structure (match your dataset_loader.py)
DATASET_FOLDER = "./dataset"
DATA_ROOMS = ["bedroom_lviv", "parents_home", "vitalnia_lviv"]
DATA_SUBROOMS = [["1", "2", "3", "4"], ["1"], ["1", "2", "3", "4", "5"]]

# LSTM Model parameters
input_dim = 468  # 114 subcarriers * 4 antenna_pairs * 2 (amplitude + phase)
hidden_dim = 256
layer_dim = 2
output_dim = 7
dropout_rate = 0.5
bidirectional = False
SEQ_DIM = 1024
DATA_STEP = 8

BATCH_SIZE = 16
EPOCHS_NUM = 100
LEARNING_RATE = 0.00146

class_weights = (
    torch.Tensor([0.113, 0.439, 0.0379, 0.1515, 0.0379, 0.1212, 0.1363])
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

def load_data():
    # List all sessions across rooms
    all_sessions = []
    for room_idx, room in enumerate(DATA_ROOMS):
        for subroom in DATA_SUBROOMS[room_idx]:
            session_path = os.path.join(DATASET_FOLDER, room, subroom)
            all_sessions.append(session_path)

    # Split sessions into train/val (80/20)
    train_sessions, val_sessions = train_test_split(
        all_sessions, test_size=0.2, random_state=42, shuffle=True
    )

    # Initialize datasets with the same parameters
    train_dataset = CSIDataset(train_sessions, window_size=SEQ_DIM, step=DATA_STEP,is_training=True)
    val_dataset = CSIDataset(val_sessions, window_size=SEQ_DIM, step=DATA_STEP,is_training=False )
    logging.info("Data is loaded...")

    trn_dl = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_dl = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return trn_dl, val_dl


# def load_data():
#     logging.info("Loading data...")

#     train_dataset = CSIDataset(
#         [
#             ".//dataset//bedroom_lviv//1",
#             ".//dataset//bedroom_lviv//2",
#             ".//dataset//bedroom_lviv//3",
#             ".//dataset//parents_home//1",
#             ".//dataset//vitalnia_lviv//1",
#             ".//dataset//vitalnia_lviv//2",
#             ".//dataset//vitalnia_lviv//3",
#             ".//dataset//vitalnia_lviv//4",        
#         ],
#         SEQ_DIM,
#         DATA_STEP,
#     )

#     val_dataset = CSIDataset([
#         ".//dataset//bedroom_lviv//4",
#         ".//dataset//vitalnia_lviv//5"
#     ], SEQ_DIM)

#     logging.info("Data is loaded...")

#     trn_dl = DataLoader(
#         train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
#     )
#     val_dl = DataLoader(
#         val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
#     )

#     return trn_dl, val_dl


def train():
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # L2 regularization
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5,patience=5,)

    # Load checkpoint if available
    checkpoint_filename = "checkpoint.pth"
    try:
        checkpoint = load_checkpoint(checkpoint_filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
    except FileNotFoundError:
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

        train_accuracy = correct_predictions / total_predictions

        val_loss, val_correct, val_total, val_acc = get_train_metric(
            model, val_dl, criterion, BATCH_SIZE
        )
        # train_loss, train_correct, train_total, train_acc = get_train_metric(model, trn_dl, criterion, BATCH_SIZE)

        logging.info(
            f"Epoch: {epoch:3d} |"
            f" Validation Loss: {val_loss/len(val_dl):.2f}, Validation Acc.: {val_acc:2.2%}, "
            f"Train Loss: {epoch_loss / len(trn_dl):.4f}, Train Acc: {train_accuracy:.2%}"
        )

        if val_acc > best_acc:
            trials = 0
            best_acc = val_acc
            torch.save(model.state_dict(), "saved_models/lstm_classifier_best.pth")
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
