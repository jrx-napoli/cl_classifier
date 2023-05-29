import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, latent_size, n_classes, hidden_size, device):
        super().__init__()
        self.device = device
        self.fc_1 = nn.Linear(latent_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = self.fc_2(x)
        return x
