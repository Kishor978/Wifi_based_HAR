import logging
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loader import CSIDataset
from models import CNN_BiLSTM_Attention
from metrics import get_train_metric

logging.basicConfig(level=logging.INFO)

# Cuda support
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Device: {}".format(device))

# Model parameters
input_dim = 468
hidden_dim = 256
layer_dim = 2
output_dim = 7
dropout_rate = 0.0
bidirectional = False
SEQ_DIM = 1024
DATA_STEP = 8
BATCH_SIZE = 4
EPOCHS_NUM = 1
LEARNING_RATE = 0.00146

class_weights = torch.Tensor([0.113, 0.439, 0.0379, 0.1515, 0.0379, 0.1212, 0.1363]).double().to(device)
class_weights_inv = 1 / class_weights
logging.info("class_weights_inv: {}".format(class_weights_inv))


def load_data():
    logging.info("Loading data...")

    train_dataset = CSIDataset([
        ".\\dataset\\bedroom_lviv\\1",
        ".\\dataset\\bedroom_lviv\\2",
        ".\\dataset\\bedroom_lviv\\3",
        ".\\dataset\\bedroom_lviv\\4",
        ".\\dataset\\vitalnia_lviv\\1",
        ".\\dataset\\vitalnia_lviv\\2",
        ".\\dataset\\vitalnia_lviv\\3",
        ".\\dataset\\vitalnia_lviv\\4",
    ], SEQ_DIM, DATA_STEP)

    val_dataset = train_dataset

    logging.info("Data is loaded...")

    trn_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return trn_dl, val_dl


def train():
    patience, trials, best_acc = 100, 0, 0
    trn_dl, val_dl = load_data()

    # Initialize the CNN-BiLSTM-Attention model
    model = CNN_BiLSTM_Attention(input_dim, hidden_dim, layer_dim, dropout_rate, bidirectional, output_dim,seq_dim=SEQ_DIM)
    model = model.double().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_inv)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    logging.info("Start model training")
    for epoch in range(1, EPOCHS_NUM + 1):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for i, (x_batch, y_batch) in tqdm(enumerate(trn_dl), total=len(trn_dl), desc=f"Training epoch {epoch}"):
            if x_batch.size(0) != BATCH_SIZE:
                continue

            # Reshape x_batch to match CNN input (batch_size, channels, height, width)
            x_batch = x_batch.unsqueeze(1)  # Adding a channel dimension (batch_size, 1, height, width)
            x_batch, y_batch = x_batch.double().to(device), y_batch.long().to(device)

            optimizer.zero_grad()

            # Forward pass
            out = model(x_batch)

            loss = criterion(out, y_batch)
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Metrics
            _, predicted = torch.max(out, 1)
            correct_predictions += (predicted == y_batch).sum().item()
            total_predictions += y_batch.size(0)

        train_accuracy = correct_predictions / total_predictions
        val_loss, val_correct, val_total, val_acc = get_train_metric(model, val_dl, criterion, BATCH_SIZE)

        logging.info(f'Epoch: {epoch:3d} | '
                     f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}, '
                     f'Train Loss: {epoch_loss / len(trn_dl):.4f}, Train Acc: {train_accuracy:.2%}')

        # Check for model improvement
        if val_acc > best_acc:
            trials = 0
            best_acc = val_acc
            torch.save(model.state_dict(), 'saved_models/cnn_bilstm_attention_best.pth')
            logging.info(f'Best model saved with accuracy: {best_acc:.2%}')
        else:
            trials += 1
            if trials >= patience:
                logging.info(f'Early stopping on epoch {epoch}')
                break

        scheduler.step(val_loss)


if __name__ == '__main__':
    train()
