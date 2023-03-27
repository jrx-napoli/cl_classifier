import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, latent_size, device):
        super().__init__()
        self.device = device
        self.fc_1 = nn.Linear(latent_size, latent_size * 2)
        self.fc_2 = nn.Linear(latent_size * 2, 100)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, model_type, latent_size, device, in_size, in_channels=1):
        super().__init__()
        self.device = device
        self.in_size = in_size
        self.model_type = model_type
        self.latent_size = latent_size
        self.in_channels = in_channels
        
        if self.model_type == 'mlp400':
            self.fc_1 = nn.Linear(in_size * in_size * in_channels, 400)
            self.fc_2 = nn.Linear(400, 400)
            self.fc_3 = nn.Linear(400, self.latent_size)
        
        elif self.model_type == 'conv':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
            self.fc_1 = nn.Linear(3136, 1024)
            self.fc_2 = nn.Linear(1024, 512)
            self.fc_3 = nn.Linear(512, self.latent_size)


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
