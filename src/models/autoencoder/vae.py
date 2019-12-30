from typing import Tuple

import torch
from torch import nn


# class VAE(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.conv_encoder = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#
#             nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU()
#         )
#
#         self.linear_encoder = nn.Sequential(
#             nn.Linear(16 * 50 * 50, 512),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )
#
#         self.mu_encoder = nn.Linear(512, 512)
#         self.log_var_encoder = nn.Linear(512, 512)
#
#         self.linear_decoder = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.Linear(512, 8 * 8 * 16),
#             nn.BatchNorm1d(8 * 8 * 16)
#         )
#
#         self.conv_decoder = nn.Sequential(
#             nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#
#             nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#
#             nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
#         )
#
#     def forward(self, x):
#         mu, log_var = self.__encode(x)
#         z = VAE.__parameterize(mu, log_var)
#         return self.decode(z), mu, log_var
#
#     def __encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x = self.conv_encoder(x)
#         x = x.view(-1, 16 * 50 * 50)
#         x = self.linear_encoder(x)
#         return self.mu_encoder(x), self.log_var_encoder(x)
#
#     def __decode(self, x: torch.Tensor):
#         x = self.linear_decoder(x)
#         x = self.conv_decoder(x)
#         return x.view(-1, 3, 32, 32)
#
#     @staticmethod
#     def __parametrize(mu, log_var):
#         std = log_var.mul(0.5).exp_()
#         eps = std.data.new(std.size()).normal_()
#         return eps.mul(std).add_(mu)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_encoder = nn.Sequential(
            nn.Linear(in_features=200 * 200 * 3, out_features=10000),
            nn.BatchNorm2d(10000),
            nn.ReLU(),

            nn.Linear(in_features=10000, out_features=1000),
            nn.BatchNorm2d(1000),
            nn.ReLU(),

            nn.Linear(in_features=1000, out_features=100),
            nn.BatchNorm2d(100),
            nn.ReLU()
        )

        self.mu_encoder = nn.Linear(100, 1)
        self.log_var_encoder = nn.Linear(100, 1)

        self.linear_decoder = nn.Sequential(
            nn.Linear(in_features=100, out_features=1000),
            nn.BatchNorm2d(1000),
            nn.ReLU(),

            nn.Linear(in_features=1000, out_features=10000),
            nn.BatchNorm2d(10000),
            nn.ReLU(),

            nn.Linear(in_features=10000, out_features=200 * 200 * 3)
        )

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = VAE.__parameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, 3 * 200 * 200)
        x = self.linear_encoder(x)
        return self.mu_encoder(x), self.log_var_encoder(x)

    def decode(self, x: torch.Tensor):
        x = self.linear_decoder(x)
        return x.view(-1, 3, 200, 200)

    @staticmethod
    def parametrize(mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
