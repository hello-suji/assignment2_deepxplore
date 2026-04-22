import torch
import torch.nn as nn


class NeuronCoverage:
    def __init__(self, model, threshold=0.0):
        self.model = model
        self.threshold = threshold
        self.covered = {}
        self.total_neurons = 0
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, layer_name):
        def hook(module, inputs, output):
            if output.dim() == 4:
                activated = output.detach().amax(dim=(0, 2, 3)) > self.threshold
            elif output.dim() == 2:
                activated = output.detach().amax(dim=0) > self.threshold
            else:
                return

            activated = activated.cpu()

            # 🔥 핵심: shape 포함해서 key 분리
            key = f"{layer_name}_{activated.shape}"

            if key not in self.covered:
                self.covered[key] = torch.zeros_like(activated, dtype=torch.bool)
                self.total_neurons += activated.numel()

            self.covered[key] |= activated

        return hook

    def coverage_ratio(self):
        covered_count = 0
        for value in self.covered.values():
            covered_count += value.sum().item()

        if self.total_neurons == 0:
            return 0.0

        return covered_count / self.total_neurons

    def close(self):
        for handle in self.handles:
            handle.remove()