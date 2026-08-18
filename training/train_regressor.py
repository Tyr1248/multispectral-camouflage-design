# train_regressor.py — Train the 1D convolutional Lab regressor (optional alternative)
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import random
import os

from dataset import get_train_val_split, CustomDataset
from lab_regressor_conv1d import Conv1DModel


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)      # (B, 4)
        labels = labels.to(device)      # (B, 3)
        inputs = inputs.unsqueeze(1)  # (B, 4) -> (B, 1, 4) to fit Conv1d

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device).unsqueeze(1)  # (B, 1, 4)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs('weight', exist_ok=True)

    # Load data (also saves y_mean.npy / y_std.npy to parameters/)
    X_train, y_train, X_val, y_val = get_train_val_split("dataset.csv")

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=50000, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=12500, shuffle=False)

    model = Conv1DModel(input_channels=1, output_dim=3).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=5e-3)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 1.0 if step < 5000 else max(0.0, (10000 - step) / 5000)
    )

    num_epochs = 10000
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        scheduler.step()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "weight/lab_regressor.pth")
            print(f"New best model saved as: weight/lab_regressor.pth")


if __name__ == "__main__":
    main()
