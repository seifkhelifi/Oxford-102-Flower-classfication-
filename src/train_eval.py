import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import timm

from utils import plot_metrics
from oxford102 import get_flowers102_dataloader
from models import get_model


import os


def train_and_evaluate(
    model_name: str,
    root: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 0,
):
    print(f"lr : {lr} | batch size : {batch_size} |  weight decay : {weight_decay}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_flowers102_dataloader(root, batch_size, "train")
    val_loader = get_flowers102_dataloader(root, batch_size, "val", shuffle=False)

    model = get_model(model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"[{model_name.upper()}] Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        plot_metrics(
            model_name,
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            num_epochs=num_epochs,
            output_dir=output_dir,
        )


# Example usage in Colab:
if __name__ == "__main__":
    output_path = "/content/drive/MyDrive/flower_results"
    data_path = "data/flowers102"

    train_and_evaluate(
        "resnet50", data_path, output_path, num_epochs=10, batch_size=32, lr=1e-4
    )
    train_and_evaluate(
        "vit_b_16", data_path, output_path, num_epochs=20, batch_size=32, lr=1e-4
    )
    train_and_evaluate(
        "efficientnet_b0",
        data_path,
        output_path,
        num_epochs=10,
        batch_size=32,
        lr=0.0005,
    )
    train_and_evaluate(
        "vit_small_patch16_224",
        data_path,
        output_path,
        num_epochs=60,
        batch_size=64,
        lr=1e-4,
    )
