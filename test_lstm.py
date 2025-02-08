# import logging
# import torch
# from torch.utils.data import DataLoader

# from dataset_loader import CSIDataset
# from metrics import get_train_metric
# from models import LSTMClassifier
# from tqdm import tqdm

# # LSTM Model parameters
# input_dim = 468  # 114 subcarriers * 4 antenna_pairs * 2 (amplitude + phase)
# hidden_dim = 256
# layer_dim = 2
# output_dim = 7
# dropout_rate = 0.01
# bidirectional = False
# SEQ_DIM = 1024
# DATA_STEP = 8

# BATCH_SIZE = 16
# EPOCHS_NUM = 1
# LEARNING_RATE = 0.00146

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logging.basicConfig(level=logging.INFO)
# # model = InceptionModel(input_dim, hidden_dim, layer_dim, output_dim, dropout_rate, bidirectional)
# model = LSTMClassifier(input_dim, hidden_dim, layer_dim, dropout_rate, bidirectional, output_dim, BATCH_SIZE)

# # model=LSTMClassifier(3, SEQ_DIM, input_dim, 12, 15, True, output_dim)

# model.load_state_dict(state_dict=torch.load(".\\saved_models\\simple_lstm_best.pth"))
# model = model.to(device)
# model = model.double()

# model.eval()


# val_dataset = CSIDataset([
#     ".\\dataset\\vitalnia_lviv\\5",
#     # "./dataset/vitalnia_lviv/5/"
# ], 1024)

# val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# model.eval()
# for i, (x_batch, y_batch) in tqdm(enumerate(val_dl), total=len(val_dl), desc="Testing epoch: "):
#     if x_batch.size(0) != BATCH_SIZE:
#         continue

#     # model.hidden = model.init_hidden(x_batch.size(0))
#     x_batch, y_batch = x_batch.double().to(device), y_batch.double().to(device)
#     out = model.forward(x_batch)

#     print("out: ", out.shape)
#     print("Predicted Class: ", torch.argmax(torch.nn.functional.log_softmax(out, dim=1), dim=1))
#     print("Actual Class    : ", y_batch)

#     preds = torch.nn.functional.log_softmax(out, dim=1).argmax(dim=1)

import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_loader import CSIDataset
from models import LSTMClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    # Model parameters
    input_dim = 468
    hidden_dim = 256
    layer_dim = 2
    output_dim = 7
    dropout_rate = 0.01
    bidirectional = False
    SEQ_DIM = 1024
    DATA_STEP = 8

    BATCH_SIZE = 16
    EPOCHS_NUM = 1
    LEARNING_RATE = 0.00146

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(level=logging.INFO)

    model = LSTMClassifier(
        input_dim,
        hidden_dim,
        layer_dim,
        dropout_rate,
        bidirectional,
        output_dim,
        BATCH_SIZE,
    )
    model.load_state_dict(torch.load(".\\saved_models\\lstm_classifier_best_final.pth"))
    model = model.to(device).double()

    model.eval()

    val_dataset = CSIDataset([".\\dataset\\vitalnia_lviv\\5"], 1024)

    val_dl = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )  # Set num_workers to 0

    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for i, (x_batch, y_batch) in tqdm(
        enumerate(val_dl), total=len(val_dl), desc="Testing epoch: "
    ):
        if x_batch.size(0) != BATCH_SIZE:
            continue

        x_batch, y_batch = x_batch.double().to(device), y_batch.to(device)
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
