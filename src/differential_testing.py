import torch


@torch.no_grad()
def find_disagreements(model_a, model_b, loader, device, max_examples=20):
    model_a.eval()
    model_b.eval()

    disagreements = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs_a = model_a(images)
        outputs_b = model_b(images)

        preds_a = outputs_a.argmax(dim=1)
        preds_b = outputs_b.argmax(dim=1)

        diff_mask = preds_a != preds_b
        diff_indices = diff_mask.nonzero(as_tuple=False).flatten().tolist()

        for idx in diff_indices:
            disagreements.append({
                "image": images[idx].detach().cpu(),
                "true_label": labels[idx].item(),
                "pred_a": preds_a[idx].item(),
                "pred_b": preds_b[idx].item(),
            })

            if len(disagreements) >= max_examples:
                return disagreements

    return disagreements