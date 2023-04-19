import numpy as np
import torch
import torch.nn as nn

def img_conv(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(inplace=True)
            # nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(inplace=True)
            # nn.LeakyReLU(0.1, inplace=True)
        )


class ImgEncoder(nn.Module):
    def __init__(self, args, batch_norm=True):
        super(ImgEncoder, self).__init__()
        self.args = args
        self.batch_norm = batch_norm
        self.conv1 = img_conv(self.batch_norm, 6, 16, kernel_size=7, stride=2)
        self.conv2 = img_conv(self.batch_norm, 16, 32, kernel_size=5, stride=2)
        self.conv3 = img_conv(self.batch_norm, 32, 64, kernel_size=3, stride=2)
        self.conv4 = img_conv(self.batch_norm, 64, 128, kernel_size=3, stride=2)
        self.conv5 = img_conv(self.batch_norm, 128, 256, kernel_size=3, stride=2)
        self.conv6 = img_conv(self.batch_norm, 256, 256, kernel_size=3, stride=2)
        self.conv7 = img_conv(self.batch_norm, 256, 256, kernel_size=3, stride=2)
    

    def forward(self, img_pair):
        """
        Input:
        -> img_pair: stacked image pair: [batch, 6, H, W]
        Output:
        -> cnv7: [batch, 256, rH, rW]
        """
        cnv1 = self.conv1(img_pair)
        cnv2 = self.conv2(cnv1)
        cnv3 = self.conv3(cnv2)
        cnv4 = self.conv4(cnv3)
        cnv5 = self.conv5(cnv4)
        cnv6 = self.conv6(cnv5)
        cnv7 = self.conv7(cnv6)

        # kitti: [batch, 256, 3, 10] for (376, 1241)
        #        [batch, 256, ]
        # euroc: [batch, 256, 4, 6] for (480, 752)
        return cnv7

