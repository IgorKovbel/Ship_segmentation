import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, is_skip_connection=False):
        super(BasicBlock, self).__init__()

        self.relu = nn.ReLU()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = downsample
        self.is_skip_connection = is_skip_connection


    def forward(self, x):
        out = self.initial(x)

        if self.is_skip_connection:
            if self.downsample is not None:
                x = self.downsample(x)

            out += x

        out = self.relu(out)

        return out

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = self.make_layer(64, 64, 3)
        self.layer2 = self.make_layer(64, 128, 4, stride=2)
        self.layer3 = self.make_layer(128, 256, 6, stride=2)
        self.layer4 = self.make_layer(256, 512, 3, stride=2)

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample, is_skip_connection=True))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, is_skip_connection=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        e1_out = x

        x = self.maxpool(x)
        x = self.layer1(x)
        e2_out = x

        x = self.layer2(x)
        e3_out = x

        x = self.layer3(x)
        e4_out = x

        x = self.layer4(x)

        return x, e1_out, e2_out, e3_out, e4_out

class UpConv(nn.Module):
    def __init__(self, channels):
        super(UpConv, self).__init__()
        
        self.relu = nn.ReLU()
        self.identity = nn.Identity()
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.relu(x)
        x = self.identity(x)
        x = self.up(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, up_channels, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.up = UpConv(up_channels)
        self.block = BasicBlock(in_channels, out_channels)
    
    def forward(self, x, encoder_block_output=None):
        x = self.up(x)

        if encoder_block_output is not None:
            x = self.block(torch.cat([x, encoder_block_output], dim=1))
        else:
            x = self.block(x)
        return x
    
class UnetDecoder(nn.Module):
    def __init__(self):
        super(UnetDecoder, self).__init__()

        self.blocks = nn.ModuleList([
            DecoderBlock(512, 512+256, 256),
            DecoderBlock(256, 256+128, 128),
            DecoderBlock(128, 128+64, 64),
            DecoderBlock(64, 64+64, 32),
            DecoderBlock(32, 32, 16),
        ])
    
    def forward(self, x, e1_out, e2_out, e3_out, e4_out):
        x = self.blocks[0](x, e4_out)
        x = self.blocks[1](x, e3_out)
        x = self.blocks[2](x, e2_out)
        x = self.blocks[3](x, e1_out)
        x = self.blocks[4](x)

        return x

class SegmentationHead(nn.Module):
    def __init__(self):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class SegmentationModel(nn.Module):
    def __init__(self, encoder, decoder, segmentation_head):
        super(SegmentationModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.segmentation_head = segmentation_head
        self.dice_loss = DiceLoss()

    def forward(self, x):
        x, e1_out, e2_out, e3_out, e4_out = self.encoder(x)
        x = self.decoder(x, e1_out, e2_out, e3_out, e4_out)
        x = self.segmentation_head(x)
        return x

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice