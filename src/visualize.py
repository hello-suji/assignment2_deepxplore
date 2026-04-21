import os
import matplotlib.pyplot as plt
import numpy as np


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def save_disagreement_images(disagreements, save_dir="results", max_save=5):
    os.makedirs(save_dir, exist_ok=True)

    for i, item in enumerate(disagreements[:max_save]):
        image = item["image"].permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)

        true_label = CIFAR10_CLASSES[item["true_label"]]
        pred_a = CIFAR10_CLASSES[item["pred_a"]]
        pred_b = CIFAR10_CLASSES[item["pred_b"]]

        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"True: {true_label}\nModel A: {pred_a}\nModel B: {pred_b}"
        )
        plt.tight_layout()
        plt.savefig(f"{save_dir}/disagreement_{i+1}.png")
        plt.close()