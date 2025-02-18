import torch
from torch.nn import functional as F
from tqdm import tqdm

# Cuda support
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def get_train_metric(model, dl, criterion, BATCH_SIZE):
#     model.eval()

#     correct, total, total_loss = 0, 0, 0

#     model.hidden = model.init_hidden(BATCH_SIZE)
#     for x_val, y_val in tqdm(dl, total=len(dl), desc="Validation epoch: "):
#         if x_val.size(0) != BATCH_SIZE:
#             continue

#         model.init_hidden(x_val.size(0))
#         x_val, y_val = x_val.double().to(device), y_val.double().to(device)

#         out = model(x_val)

#         # out = out.view(out.size(0) * out.size(1), out.size(2))
#         # y_val = y_val.view(y_val.size(0) * y_val.size(1))
#         # print("y_val.size(0): ", y_val.size())

#         loss = criterion(out, y_val.long())

#         total_loss += loss.item()

#         preds = F.log_softmax(out, dim=1).argmax(dim=1)
#         total += y_val.size(0)
#         correct += (preds == y_val).sum().item()

#     acc = correct / total


#     return total_loss, correct, total, acc
def get_train_metric(model, dl, criterion, BATCH_SIZE):
    model.eval()  # Set model to evaluation mode

    correct, total, total_loss = 0, 0, 0

    with torch.no_grad():  # Disable gradient computation
        for x_val, y_val in tqdm(dl, total=len(dl), desc="Validation epoch: "):
            if x_val.size(0) != BATCH_SIZE:
                continue

            # Convert inputs & labels to the correct types
            x_val, y_val = x_val.to(device).float(), y_val.to(device).long()

            out = model(x_val)

            loss = criterion(out, y_val)
            total_loss += loss.item()

            preds = out.argmax(dim=1)  # Directly use argmax instead of log_softmax + argmax
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()

    acc = correct / total

    return total_loss, correct, total, acc


def get_train_metric_BiLSTM(model, val_dl, criterion, batch_size):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for x_batch, y_batch in tqdm(
            val_dl, total=len(val_dl), desc="Validation epoch: "
        ):
            if x_batch.size(0) != batch_size:
                continue

            # Unsqueeze to add channel dimension
            x_batch = x_batch.unsqueeze(1)

            x_batch, y_batch = x_batch.double().to(device), y_batch.long().to(device)

            # Forward pass
            out = model(x_batch)

            loss = criterion(out, y_batch)
            val_loss += loss.item()

            _, predicted = torch.max(out, 1)
            correct_predictions += (predicted == y_batch).sum().item()
            total_predictions += y_batch.size(0)

    val_accuracy = correct_predictions / total_predictions
    avg_val_loss = val_loss / len(val_dl)

    return avg_val_loss, correct_predictions, total_predictions, val_accuracy
