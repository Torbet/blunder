import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt

# models
from models.dense import Dense1, Dense3, Dense6  # stanford paper
from models.conv import Conv1, Conv3, Conv6  # stanford paper
from models.conv_lstm import ConvLSTM, ConvLSTMExtra
from models.conv3d import Conv3D

# parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 1e-4
epochs = 20
limit = 10000
data_path = f"data/processed/{limit}"
torch.random.manual_seed(42)


# dataset
class Dataset(data.Dataset):
    def __init__(self, dir: str):
        self.moves = np.load(f"{dir}/moves.npy").astype(np.float32)
        self.evals = np.load(f"{dir}/evals.npy").astype(np.float32)
        self.times = np.nan_to_num(np.load(f"{dir}/times.npy").astype(np.float32))
        self.labels = np.load(f"{dir}/labels.npy").astype(np.float32)

        # shuffle
        idx = np.random.permutation(len(self))
        self.moves = self.moves[idx]
        self.evals = self.evals[idx]
        self.times = self.times[idx]
        self.labels = self.labels[idx]

    def __len__(self):
        return self.moves.shape[0]

    def __getitem__(self, idx):
        return self.moves[idx], self.evals[idx], self.times[idx], self.labels[idx]


def load_data(path: str):
    # Load dataset
    dataset = Dataset(path)

    # Split dataset
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = test_size = (n - train_size) // 2

    train_dataset, val_dataset, test_dataset = data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train(model: nn.Module, train_loader: data.DataLoader, val_loader: data.DataLoader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses, accuracies = [], []

    for epoch in range(epochs):
        # train
        model.train()
        epoch_loss = 0
        for moves, evals, times, labels in train_loader:
            moves, evals, times, labels = (
                moves.to(device),
                evals.to(device),
                times.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            predicted = model(moves, evals, times)
            loss = criterion(predicted, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))

        # test
        accuracy = evaluate(model, val_loader)
        accuracies.append(accuracy)

        # log
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {losses[-1]:.2f}, Accuracy: {(accuracies[-1]*100):.2f}%"
        )

    return losses, accuracies


def evaluate(model: nn.Module, loader: data.DataLoader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for moves, evals, times, labels in loader:
            moves, evals, times, labels = (
                moves.to(device),
                evals.to(device),
                times.to(device),
                labels.to(device),
            )
            predicted = model(moves, evals, times)
            _, predicted = torch.max(predicted, 1)
            _, labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


if __name__ == "__main__":
    model = Dense1().to(device)
    model_name = model.__class__.__name__
    print(model_name, "\n")
    train_loader, val_loader, test_loader = load_data(data_path)
    losses, accuracies = train(model, train_loader, val_loader)
    test_accuracy = evaluate(model, test_loader)
    print(f"\nTest accuracy: {(test_accuracy*100):.2f}%")
    np.savez(f"results/{model_name}-{limit}.npz", losses=losses, accuracies=accuracies)
    plt.plot(losses)
    plt.plot(accuracies)
    plt.show()
