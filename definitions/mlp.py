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


class MLP(nn.Module):
    def __init__(self, model_type, latent_size, device, in_size, hidden_size=400, in_channels=1):
        super().__init__()
        self.device = device
        self.model_type = model_type

        if self.model_type == 'mlp400':
            self.fc_1 = nn.Linear(in_size * in_size * in_channels, hidden_size)
            self.fc_2 = nn.Linear(hidden_size, hidden_size)
            self.fc_3 = nn.Linear(hidden_size, latent_size)

        elif self.model_type == 'conv':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
            self.fc_1 = nn.Linear(3136, 1024)
            self.fc_2 = nn.Linear(1024, 512)
            self.fc_3 = nn.Linear(512, latent_size)

    def forward(self, x):
        if self.model_type == 'mlp400':
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.fc_1(x))
            x = F.leaky_relu(self.fc_2(x))
            x = F.leaky_relu(self.fc_3(x))

        elif self.model_type == 'conv':
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
