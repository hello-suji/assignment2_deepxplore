import torch.nn as nn
from torchvision.models import resnet50


def get_resnet50_cifar10():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model