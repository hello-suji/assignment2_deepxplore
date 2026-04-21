import os
import random
import numpy as np
import torch
import torch.optim as optim

from src.data_utils import get_cifar10_loaders
from src.models import get_resnet50_cifar10
from src.train import train_one_epoch, evaluate
from src.coverage import NeuronCoverage
from src.differential_testing import find_disagreements
from src.visualize import save_disagreement_images


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(model, train_loader, test_loader, device, lr, seed, save_path, epochs=1):
    set_seed(seed)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        print(
            f"[Seed={seed} | LR={lr}] "
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss={loss:.4f} | Train Acc={train_acc:.4f} | Test Acc={test_acc:.4f}"
        )

    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


def load_or_train_models(train_loader, test_loader, device):
    os.makedirs("checkpoints", exist_ok=True)

    model_a = get_resnet50_cifar10()
    model_b = get_resnet50_cifar10()

    path_a = "checkpoints/model_a.pth"
    path_b = "checkpoints/model_b.pth"

    if os.path.exists(path_a):
        model_a.load_state_dict(torch.load(path_a, map_location=device))
        print("Loaded model A from checkpoint.")
    else:
        train_model(
            model=model_a,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            lr=1e-3,
            seed=42,
            save_path=path_a,
            epochs=1,
        )

    if os.path.exists(path_b):
        model_b.load_state_dict(torch.load(path_b, map_location=device))
        print("Loaded model B from checkpoint.")
    else:
        train_model(
            model=model_b,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            lr=5e-4,
            seed=123,
            save_path=path_b,
            epochs=1,
        )

    model_a.to(device)
    model_b.to(device)
    return model_a, model_b


def measure_coverage(model, test_loader, device, num_batches=10):
    nc = NeuronCoverage(model, threshold=0.0)
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(test_loader):
            images = images.to(device)
            model(images)

            if batch_idx + 1 >= num_batches:
                break

    ratio = nc.coverage_ratio()
    nc.close()
    return ratio


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_cifar10_loaders(batch_size=32)

    model_a, model_b = load_or_train_models(train_loader, test_loader, device)

    coverage_a = measure_coverage(model_a, test_loader, device)
    coverage_b = measure_coverage(model_b, test_loader, device)

    print(f"Neuron coverage (Model A): {coverage_a:.4f}")
    print(f"Neuron coverage (Model B): {coverage_b:.4f}")

    disagreements = find_disagreements(
        model_a=model_a,
        model_b=model_b,
        loader=test_loader,
        device=device,
        max_examples=20,
    )

    print(f"Found {len(disagreements)} disagreement-inducing inputs.")

    save_disagreement_images(disagreements, save_dir="results", max_save=5)
    print("Saved at least 5 disagreement images to results/ directory.")

    with open("results/summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Neuron coverage (Model A): {coverage_a:.4f}\n")
        f.write(f"Neuron coverage (Model B): {coverage_b:.4f}\n")
        f.write(f"Number of disagreements found: {len(disagreements)}\n")


if __name__ == "__main__":
    main()