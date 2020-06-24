from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

class Conv(nn.Module):

  def __init__(self, in_c, out_c, kernel_size = 3, dilation = 1, padding = 1):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size = kernel_size, dilation = dilation, padding = padding),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace = True),
    ) 

  def forward(self,x):
    out = self.conv(x)
    return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(UpBlock, self).__init__()

        self.up = nn.Sequential(
        nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True))
       
        self.conv_block1 = Conv(out_channels*2 ,out_channels) 
        self.conv_block2 = Conv(out_channels ,out_channels) 

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block1(out)
        out = self.conv_block2(out)
        return out

class UNet(nn.Module):
    def __init__(self,pretrained=True):

        super().__init__()

        self.encoder = models.vgg13_bn(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace = True)
        self.pool = self.encoder[6]
        self.conv1 = self.encoder[0]
        
        self.conv1s = self.encoder[3]
        
        self.conv2 = self.encoder[7]
        self.conv2s = self.encoder[10]
        self.conv3 = self.encoder[14]
        self.conv3s = self.encoder[17]
        self.conv4 = self.encoder[21]
        self.conv4s = self.encoder[24]
        self.bn1=self.encoder[1]
        self.bn1s=self.encoder[4]
        self.bn2=self.encoder[8]
        self.bn2s=self.encoder[11]
        self.bn3=self.encoder[15]
        self.bn3s=self.encoder[18]
        self.bn4=self.encoder[22]
        self.bn4s=self.encoder[25]


        self.center1 = Conv(512, 512, kernel_size = 3, dilation=1, padding=1)
        self.center2 = Conv(512, 512, kernel_size = 3, dilation=2, padding=2)
        self.center3 = Conv(512, 512, kernel_size = 3, dilation=4, padding=4)
        self.center4 = Conv(512, 512, kernel_size = 3, dilation=8, padding=8)

        self.dec4 = UpBlock(512,512) 
        self.dec3 = UpBlock(512,256)
        self.dec2 = UpBlock(256,128)
        self.dec1 = UpBlock(128,64)

        self.final = nn.Conv2d(64, 2, kernel_size=1)
      

    def forward(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv1s = self.relu(self.bn1s(self.conv1s(conv1)))
        conv2 = self.relu(self.bn2(self.conv2(self.pool(conv1s))))
        conv2s = self.relu(self.bn2s(self.conv2s(conv2)))
        conv3 = self.relu(self.bn3(self.conv3(self.pool(conv2s))))
        conv3s = self.relu(self.bn3s(self.conv3s(conv3)))
        conv4 = self.relu(self.bn4(self.conv4(self.pool(conv3s))))
        conv4s = self.relu(self.bn4s(self.conv4s(conv4)))
       
        dilated = []
        center1 = self.center1(self.pool(conv4s))
        dilated.append(center1.unsqueeze(-1))
        center2 = self.center2(center1)
        dilated.append(center2.unsqueeze(-1))
        center3 = self.center3(center2)
        dilated.append(center3.unsqueeze(-1))
        center4 = self.center4(center3)
        dilated.append(center4.unsqueeze(-1))

        centers = torch.cat(dilated, dim=-1)
        center = torch.sum(centers, dim=-1)

        dec4 = self.dec4(center,conv4s)
        dec3 = self.dec3(dec4,conv3s)
        dec2 = self.dec2(dec3,conv2s)
        dec1 = self.dec1(dec2,conv1s)
        return self.final(dec1)