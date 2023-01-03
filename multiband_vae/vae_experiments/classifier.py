import torch.nn as nn
import torch.nn.functional as F
import torch

from vae_experiments.vae_utils import BitUnpacker

class Classifier(nn.Module):
    def __init__(self, latent_size, d, cond_p_coding, cond_n_dim_coding, cond_dim, device, in_size, fc):
        super().__init__()

        self.feature_extractor = FeatureExtractor(latent_size, d, cond_dim, cond_p_coding, cond_n_dim_coding, device, in_size, fc)
        self.head = Head(latent_size, d, cond_dim, cond_p_coding, cond_n_dim_coding, device, in_size, fc)

    def forward(self):
        pass

class Head(nn.Module):
    def __init__(self, latent_size, d, device, in_size, fc):
        super().__init__()
        self.d = d
        self.device = device
        self.in_size = in_size
        self.only_fc = fc

        # self.fc_1 = nn.Linear(self.d * latent_size, self.d * 8)
        # self.fc_2 = nn.Linear(self.d * 8, self.d * 4)
        # self.fc_3 = nn.Linear(self.d * 4, 10)

        self.fc_1 = nn.Linear(100, 75)
        self.fc_2 = nn.Linear(75, 50)
        self.fc_3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = self.fc_3(x)
        return x


class FeatureExtractor(nn.Module):

    def __init__(self, latent_size, d, cond_dim, cond_p_coding, cond_n_dim_coding, device, in_size, fc):
        super().__init__()
        self.d = d
        self.p = 0.6
        self.cond_p_coding = cond_p_coding
        self.cond_n_dim_coding = cond_n_dim_coding
        self.cond_dim = cond_dim
        self.device = device
        self.in_size = in_size
        self.only_fc = fc

        if self.in_size == 28:
            in_channels = 1
            self.scaler = 4
        elif self.in_size == 44:
            in_channels = 1
            self.scaler = in_size // 8
        else:
            in_channels = 3
            self.scaler = in_size // 8

        if self.only_fc:
            self.fc_1 = nn.Linear(in_size * in_size * in_channels + cond_n_dim_coding,
                                  self.d * self.scaler * self.scaler)
            self.fc_2 = nn.Linear(self.d * self.scaler * self.scaler, self.d * self.scaler * self.scaler)
            self.fc_3 = nn.Linear(self.d * self.scaler * self.scaler, self.d * latent_size) # size of translators output
            
        # todo: implementation for conv. layers
        else:
            if self.in_size == 28:
                self.dropout = nn.Dropout(self.p)
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=1, bias=True)
                self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1, bias=True)
                self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=1, bias=True)
                self.fc1 = nn.Linear(1080, 728)
                self.fc2 = nn.Linear(728, 512)
                self.fc3 = nn.Linear(512, 100)

                # self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.d, kernel_size=4, stride=2, padding=1, bias=False)
                # self.bn_1 = nn.BatchNorm2d(self.d)
                # self.conv2 = nn.Conv2d(self.d, self.d * 2, kernel_size=4, stride=2, padding=1, bias=False)
                # self.bn_2 = nn.BatchNorm2d(self.d * 2)
                # self.conv3 = nn.Conv2d(self.d * 2, self.d * 2, kernel_size=4, stride=2, padding=1, bias=False)
                # self.bn_3 = nn.BatchNorm2d(self.d * 2)
                # self.fc1 = nn.Linear(self.d * 9 * 2, self.d * 9 * 2)
                # self.fc2 = nn.Linear(self.d * 9 * 2, self.d * latent_size)


            elif self.in_size == 44:
                self.conv_out_size = 2
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.d, kernel_size=4, stride=2, padding=1,
                                       bias=False)
                self.bn_1 = nn.BatchNorm2d(self.d)
                
                self.conv2 = nn.Conv2d(self.d, self.d, kernel_size=4, stride=2, padding=1, bias=False)
                self.bn_2 = nn.BatchNorm2d(self.d)
                
                self.conv3 = nn.Conv2d(self.d, self.d * 2, kernel_size=4, stride=2, padding=1, bias=False)
                self.bn_3 = nn.BatchNorm2d(self.d * 2)
                
                self.conv4 = nn.Conv2d(self.d * 2, self.d * 4, kernel_size=4, stride=2, padding=1, bias=False)
                self.bn_4 = nn.BatchNorm2d(self.d * 4)
                self.fc = nn.Linear(self.d * 4 * self.conv_out_size * self.conv_out_size + cond_n_dim_coding,
                                    self.d * 4)

            else:
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.d, kernel_size=5, stride=2, padding=1,
                                       bias=False)
                self.bn_1 = nn.BatchNorm2d(self.d)

                self.conv2 = nn.Conv2d(self.d, self.d * 2, kernel_size=5, stride=2, padding=1, bias=False)
                self.bn_2 = nn.BatchNorm2d(self.d * 2)

                self.conv3 = nn.Conv2d(self.d * 2, self.d * 4, kernel_size=5, stride=2, padding=1, bias=False)
                self.bn_3 = nn.BatchNorm2d(self.d * 4)

                self.conv4 = nn.Conv2d(self.d * 4, self.d * 4, kernel_size=5, stride=2, padding=1, bias=False)
                self.bn_4 = nn.BatchNorm2d(self.d * 4)
                
                if self.in_size == 64:
                    self.conv_out_size = 3
                elif self.in_size == 128:
                    self.conv_out_size = 7
                else:
                    print(f"No model definition for size: {self.in_size}")
                    raise NotImplementedError
                
                self.fc1 = nn.Linear(self.d * 4 * self.conv_out_size * self.conv_out_size + cond_n_dim_coding,
                                    self.d * 4)

            # regular autoencoder instead of vae
            # self.linear = nn.Linear(self.d * 4, latent_size * self.d)


    def forward(self, x):

        if self.only_fc:
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.fc_1(x))
            x = F.leaky_relu(self.fc_2(x))
            x = F.leaky_relu(self.fc_3(x))
        else:
            # x = self.conv1(x)
            # x = F.leaky_relu(self.bn_1(x))
            # x = self.conv2(x)
            # x = F.leaky_relu(self.bn_2(x))
            # x = self.conv3(x)
            # x = F.leaky_relu(self.bn_3(x))

            x = F.leaky_relu(self.conv1(x))
            x = F.avg_pool2d(x, (2, 2))
            x = F.leaky_relu(self.conv2(x))
            x = F.avg_pool2d(x, (2, 2))
            x = F.leaky_relu(self.conv3(x))

            # print(f'x: {x.size()}')
            
            # flatten
            if self.in_size == 28:
                x = x.view([-1, 1080])
            else:
                pass

            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            x = F.leaky_relu(self.fc3(x))
        
        return x
