import torch
import torch.nn as nn
from training.networks_stylegan2 import Discriminator
from training.networks_stylegan2 import Conv2dLayer

class Discriminator_pt(nn.Module):
  def __init__(self, c_dim, resolution, channels):
    super().__init__()
    self.c_dim = c_dim
    self.resolution = resolution
    self.channels = channels
    self.D = self.load_discriminator()

  def load_discriminator(self):
    D_orig = Discriminator(self.c_dim, self.resolution, self.channels)
    D_orig.b256.fromrgb = Conv2dLayer(in_channels=3, out_channels=64, kernel_size=1,  activation='lrelu', up=1, down=1)
    D_orig.b256.conv0 = Conv2dLayer(in_channels=64, out_channels=64, kernel_size=3,  activation='lrelu', up=1, down=1)
    D_orig.b256.conv1 = Conv2dLayer(in_channels=64, out_channels=128, kernel_size=3,  activation='lrelu', up=1, down=2)
    D_orig.b128.fromrgb = Conv2dLayer(in_channels=3, out_channels=128, kernel_size=1,  activation='lrelu', up=1, down=1)
    D_orig.b128.conv0 = Conv2dLayer(in_channels=128, out_channels=128, kernel_size=3,  activation='lrelu', up=1, down=1)
    D_orig.b128.conv1 = Conv2dLayer(in_channels=128, out_channels=256, kernel_size=3,  activation='lrelu', up=1, down=2)
    D_orig.b64.fromrgb = Conv2dLayer(in_channels=3, out_channels=256, kernel_size=1,  activation='lrelu', up=1, down=1)
    D_orig.b64.conv0 = Conv2dLayer(in_channels=256, out_channels=256, kernel_size=3,  activation='lrelu', up=1, down=1)
    D_orig.b64.conv1 = Conv2dLayer(in_channels=256, out_channels=512, kernel_size=3,  activation='lrelu', up=1, down=2)
    return D_orig

  def forward(self, x):
    out = self.D(x, None)
    return out
