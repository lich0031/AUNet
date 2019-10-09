import torch
import torch.nn as nn

class Channel_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def UNet_up_conv_bn_relu(input_channel, output_channel, learned_bilinear=False):

    if learned_bilinear:
        return nn.Sequential(nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU())
    else:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                             nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
                             nn.BatchNorm2d(output_channel),
                             nn.ReLU())

class basic_block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class residual_block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        residual = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        out = self.relu(x + residual)
        return out

class UNet_residual_down_block(nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_residual_down_block, self).__init__()
        self.block = residual_block(input_channel, output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.block(x)
        return x

#################### Attention_UP_Net ##############################

class Attention_Up_block(nn.Module):
    def __init__(self, input_channel, prev_channel, output_channel, learned_bilinear=False):
        super(Attention_Up_block, self).__init__()
        self.bilinear_up = UNet_up_conv_bn_relu(input_channel, prev_channel, learned_bilinear)
        self.add_channel_conv = nn.Sequential(nn.Conv2d(input_channel, input_channel*2, kernel_size=3, padding=1),
                                              nn.BatchNorm2d(input_channel*2),
                                              nn.ReLU())
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.smooth_conv = nn.Sequential(nn.Conv2d(prev_channel, prev_channel, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(prev_channel),
                                         nn.ReLU())
        self.block = basic_block(prev_channel*2, output_channel)
        self.CA = Channel_Attention(prev_channel*2, reduction=16)

    def forward(self, pre_feature_map, x):
        x_bilinear = self.bilinear_up(x)
        x_ps = self.pixel_shuffle(self.add_channel_conv(x))
        x = self.smooth_conv((pre_feature_map + x_ps))
        x = self.CA(torch.cat((x, x_bilinear), dim=1))
        x = self.block(x)
        return x


class AUNet_R16(nn.Module):
    def __init__(self, colordim=3, n_classes=2, learned_bilinear=False):
        super(AUNet_R16, self).__init__()

        self.down_block1 = UNet_residual_down_block(colordim, 64, False)
        self.down_block2 = UNet_residual_down_block(64, 128, True)
        self.down_block3 = UNet_residual_down_block(128, 256, True)
        self.down_block4 = UNet_residual_down_block(256, 512, True)
        self.down_block5 = UNet_residual_down_block(512, 1024, True)

        self.up_block1 = Attention_Up_block(1024, 512, 512, learned_bilinear)
        self.up_block2 = Attention_Up_block(512, 256, 256, learned_bilinear)
        self.up_block3 = Attention_Up_block(256, 128, 128, learned_bilinear)
        self.up_block4 = Attention_Up_block(128, 64, 64, learned_bilinear)

        self.last_conv1 = nn.Conv2d(64, n_classes, 1, padding=0)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)

        x = self.up_block1(x4, x5)
        x = self.up_block2(x3, x)
        x = self.up_block3(x2, x)
        x = self.up_block4(x1, x)

        x = self.last_conv1(x)
        return x