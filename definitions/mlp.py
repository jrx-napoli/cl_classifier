import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, latent_size, in_size, hidden_size=400, in_channels=1):
        super().__init__()

        self.fc_1 = nn.Linear(in_size * in_size * in_channels, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_3 = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))

        return x
