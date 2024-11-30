import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import tqdm

# models
from models.dense import Dense1, Dense3, Dense6  # stanford paper
from models.conv import Conv1, Conv3, Conv6  # stanford paper
from models.conv_lstm import ConvLSTM, ConvLSTMExtra, ConvLSTMExtra2
from models.conv3d import Conv3D
from models.transformer import Swin3D

# parameters
model = ConvLSTMExtra(bidirectional=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 1e-3
epochs = 20
limit = 10000
data_path = f"data/processed/{limit}"
torch.random.manual_seed(42)
np.random.seed(42)


# dataset
class Dataset(data.Dataset):
    def __init__(self, dir: str):
        self.moves = np.load(f"{dir}/moves.npy")
        self.evals = np.load(f"{dir}/evals.npy")
        self.times = np.nan_to_num(np.load(f"{dir}/times.npy"))
        self.best_moves = np.load(f"{dir}/best_moves.npy")
        self.labels = np.load(f"{dir}/labels.npy")

        # shuffle
        idx = np.random.permutation(len(self))
        self.moves = self.moves[idx]
        self.evals = self.evals[idx]
        self.times = self.times[idx]
        self.best_moves = self.best_moves[idx]
        self.labels = self.labels[idx]

    def __len__(self):
        return self.moves.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.moves[idx], dtype=torch.float32),
            torch.tensor(self.evals[idx], dtype=torch.float32),
            torch.tensor(self.times[idx], dtype=torch.float32),
            torch.tensor(self.best_moves[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def load_data(path: str):
    # Load dataset
    dataset = Dataset(path)

    # Split dataset
    n = len(dataset)
    sizes = [int(0.8 * n), (n - int(0.8 * n)) // 2, (n - int(0.8 * n)) // 2]
    train_ds, val_ds, test_ds = data.random_split(dataset, sizes)

    # Create dataloaders
    return (
        data.DataLoader(ds, batch_size=batch_size, shuffle=(ds is train_ds), num_workers=4)
        for ds in [train_ds, val_ds, test_ds]
    )


def train(model: nn.Module, train_loader: data.DataLoader, val_loader: data.DataLoader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    losses, accuracies = [], []

    for epoch in range(epochs):
        # train
        model.train()
        epoch_loss = 0
        with tqdm(train_loader) as t:
            for moves, evals, times, best_moves, labels in train_loader:
                moves, evals, times, best_moves, labels = (
                    moves.to(device),
                    evals.to(device),
                    times.to(device),
                    best_moves.to(device),
                    labels.to(device),
                )

                optimizer.zero_grad()
                predicted = model(moves, evals, times, best_moves)
                loss = criterion(predicted, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_acc = (predicted.argmax(dim=1) == labels).sum().item() / len(labels)
                t.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.2f}, Acc: {train_acc:.2f}")
                t.update(1)
        losses.append(epoch_loss / len(train_loader))

        # test
        accuracy = evaluate(model, val_loader)
        accuracies.append(accuracy)

    return losses, accuracies


def evaluate(model: nn.Module, loader: data.DataLoader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for moves, evals, times, best_moves, labels in loader:
            moves, evals, times, best_moves, labels = (
                moves.to(device),
                evals.to(device),
                times.to(device),
                best_moves.to(device),
                labels.to(device),
            )
            predicted = model(moves, evals, times, best_moves).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


if __name__ == "__main__":
    model_name = model.__class__.__name__
    model = model.to(device)
    print(model_name, "lr:", learning_rate, "\n")
    train_loader, val_loader, test_loader = load_data(data_path)
    losses, accuracies = train(model, train_loader, val_loader)
    test_accuracy = evaluate(model, test_loader)
    print(f"\nTest accuracy: {(test_accuracy*100):.2f}%")

    # save results
    np.savez(f"results/{model_name}-{limit}.npz", losses=losses, accuracies=accuracies)

    # plot results
    plt.plot(losses, label="Loss")
    plt.plot(accuracies, label="Accuracy")
    plt.legend()
    plt.show()

    # save model
    torch.save(model.state_dict(), f"weights/{model_name}-{limit}.pt")
