import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loader import CSIDataset
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from self_utils import read_csi_data_from_csv, read_labels_from_csv
from Attention import CNN_BiLSTM_Attention

DATASET_FOLDER=".\\preprocessing\\test_data"

def read_all_data_from_files(data_path, label_path,  antenna_pairs=1):
    """
    Read CSI and labels from merged CSV files.
    """
    amplitudes, phases = read_csi_data_from_csv(data_path,  antenna_pairs)
    labels = read_labels_from_csv(label_path, len(amplitudes))
    return amplitudes, phases, labels

def main():
    input_dim = 64  
    hidden_dim = 512
    layer_dim = 2
    output_dim = 4
    dropout_rate = 0.4
    bidirectional = False
    SEQ_DIM = 1024
    DATA_STEP = 1

    BATCH_SIZE = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(level=logging.INFO)

    # Model initialization
    model = CNN_BiLSTM_Attention(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
        output_dim=output_dim,
        seq_dim=SEQ_DIM
    ).to(device)
    model.load_state_dict(torch.load(".\\saved_models\\best51.73%.pth"))
    model = model.to(device).double()

    model.eval()

    # Load merged CSI data and labels
    data_path = os.path.join(DATASET_FOLDER, "test_data.csv")
    label_path = os.path.join(DATASET_FOLDER, "test_labels.csv")
    amplitudes, phases, labels = read_all_data_from_files(data_path, label_path)
    csi_data = np.hstack((amplitudes, phases))

# Create datasets
    val_dataset = CSIDataset(
        csi_data, labels, window_size=SEQ_DIM, step=DATA_STEP, is_training=False
    )
    
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    print(val_dl)
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for i, (x_batch, y_batch) in tqdm(
        enumerate(val_dl), total=len(val_dl), desc="Testing epoch: "
    ):
        if x_batch.size(0) != BATCH_SIZE:
            continue

        x_batch = x_batch.unsqueeze(1).double().to(device)  # Ensure correct shape
        y_batch = y_batch.long().to(device)

        out = model(x_batch)

        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

        correct_predictions += (preds == y_batch).sum().item()
        total_samples += y_batch.size(0)

    accuracy = correct_predictions / total_samples
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(output_dim))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    main()
