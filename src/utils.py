import matplotlib.pyplot as plt
import os


def plot_metrics(
    model_name: str,
    train_losses: list,
    val_losses: list,
    train_accuracies: list,
    val_accuracies: list,
    output_dir: str,
    num_epochs: int,
):
    # Plot and save
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{model_name.upper()} - Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Acc")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title(f"{model_name.upper()} - Accuracy Curve")

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{model_name}_metrics.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved plot to: {plot_path}")
