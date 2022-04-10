from audioop import bias
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_experiments.vae_utils import BitUnpacker

class Classifier(nn.Module):
    def __init__(self, latent_size, binary_latent_size, d, p_coding, n_dim_coding, cond_p_coding, cond_n_dim_coding,
                 cond_dim, device, in_size, fc, standard_embeddings=False,
                 trainable_embeddings=False):
        super().__init__()

        self.feature_extractor = FeatureExtractor(latent_size, binary_latent_size, d, cond_dim, cond_p_coding, cond_n_dim_coding, device,
                               in_size, fc, standard_embeddings)

    def forward(self):
        pass


class FeatureExtractor(nn.Module):

    def __init__(self, latent_size, binary_latent_size, d, n_dim_coding, cond_dim, cond_p_coding, cond_n_dim_coding, device, in_size, fc, standard_embeddings):
        super().__init__()
        self.d = d
        self.cond_p_coding = cond_p_coding
        self.cond_n_dim_coding = cond_n_dim_coding
        self.cond_dim = cond_dim
        self.device = device
        self.in_size = in_size
        self.only_fc = fc
        self.standard_embeddings = standard_embeddings

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
            self.fc_2 = nn.Linear(self.d * self.scaler * self.scaler, self.d * self.scaler)
            self.fc_3 = nn.Linear(self.d * self.scaler, self.d * self.scaler // 2)

            # self.linear_means = nn.Linear(self.d * self.scaler // 2, latent_size)
            # self.linear_log_var = nn.Linear(self.d * self.scaler // 2, latent_size)
            # self.linear_binary = nn.Linear(self.d * self.scaler // 2, binary_latent_size)
        
            # ae instead of vae
            self.linear = nn.Linear(self.d * self.scaler // 2, latent_size)
            
            # binary latent not needed
            # self.linear_binary = nn.Linear(self.d * self.scaler // 2, binary_latent_size, bias=False)
        
        else:
            if self.in_size == 28:
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.d, kernel_size=4, stride=2, padding=1,
                                       bias=False)
                self.bn_1 = nn.BatchNorm2d(self.d)
                self.conv2 = nn.Conv2d(self.d, self.d, kernel_size=4, stride=2, padding=1, bias=False)
                self.bn_2 = nn.BatchNorm2d(self.d)
                self.conv3 = nn.Conv2d(self.d, self.d, kernel_size=4, stride=2, padding=1, bias=False)
                self.bn_3 = nn.BatchNorm2d(self.d)
                self.fc = nn.Linear(self.d * 9 + cond_n_dim_coding, self.d * 4)

            if self.in_size == 44:
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

            # self.linear_means = nn.Linear(self.d * 4, latent_size)
            # self.linear_log_var = nn.Linear(self.d * 4, latent_size)
            # self.linear_binary = nn.Linear(self.d * 4, binary_latent_size, bias=False)

            # ae instead of vae
            self.linear = nn.Linear(self.d * 4, latent_size)

            # binary latent not needed
            # self.lienar_binary = nn.Linear(self.d * 4, binary_latent_size, bias=False)

        if standard_embeddings:
            # not implemented
            # self.fc4 = nn.Linear(n_dim_coding, n_dim_coding * 2)
            # self.fc5 = nn.Linear(n_dim_coding * 2, n_dim_coding * 3)
            # self.fc6 = nn.Linear(n_dim_coding * 3, n_dim_coding)
            pass
        else:
            self.fc_enc_1 = nn.Linear(n_dim_coding, n_dim_coding * 3)
            self.fc_enc_2 = nn.Linear(n_dim_coding * 3, n_dim_coding * 2)   

            if binary_latent_size > 0:
                self.fc_bin_enc_1 = nn.Linear(binary_latent_size, binary_latent_size * 2)
                self.fc_bin_enc_2 = nn.Linear(binary_latent_size * 2, binary_latent_size * 3)

            self.fc4 = nn.Linear(n_dim_coding * 2 + latent_size + binary_latent_size * 3, latent_size * self.d // 2)
            self.fc5 = nn.Linear(latent_size * self.d // 2, latent_size * self.d)


    def forward(self, x, conds):
        with torch.no_grad():
            if self.cond_n_dim_coding:
                conds_coded = (conds * self.cond_p_coding) % (2 ** self.cond_n_dim_coding)
                conds_coded = BitUnpacker.unpackbits(conds_coded, self.cond_n_dim_coding).to(self.device)
                x = torch.cat([x, conds_coded], dim=1)
        if self.only_fc:
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.fc_1(x))
            x = F.leaky_relu(self.fc_2(x))
            x = F.leaky_relu(self.fc_3(x))
        else:
            x = self.conv1(x)
            x = F.leaky_relu(self.bn_1(x))
            x = self.conv2(x)
            x = F.leaky_relu(self.bn_2(x))
            x = self.conv3(x)
            x = F.leaky_relu(self.bn_3(x))
            if self.in_size == 28:
                x = x.view([-1, self.d * 9])
            else:
                x = self.conv4(x)
                x = F.leaky_relu(self.bn_4(x))
                x = x.view([-1, self.d * 4 * self.conv_out_size * self.conv_out_size])

            if self.cond_n_dim_coding:
                x = torch.cat([x, conds_coded], dim=1)

            x = F.leaky_relu(self.fc(x))

        # means = self.linear_means(x)
        # log_vars = self.linear_log_var(x)
        # binary_out = self.linear_binary(x)
        
        # translator like dimensional reduction
        if self.standard_embeddings:
            pass # 
        else:
            # podstawowo bez embeddingow
            pass


        return means, log_vars, binary_out

