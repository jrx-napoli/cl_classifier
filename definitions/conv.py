import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc_1 = nn.Linear(3136, 1024)
        self.fc_2 = nn.Linear(1024, 512)
        self.fc_3 = nn.Linear(512, latent_size)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))

        # flatten
        x = x.view([-1, 3136])
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))

        return x
