from typing import Iterable, Optional

import torch
import torch.nn as nn
import torchvision


class MonoLayoutEncoder(nn.Module):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        layer_sizes: Optional[Iterable] = None,
        num_resnet_layers=18,
        resnet_pretrained=True,
    ):
        super(MonoLayoutEncoder, self).__init__()
        self.resnet = ResNetFeatureExtractor(num_resnet_layers, resnet_pretrained)
        resnet_out_channels = 512
        self.conv1 = PadAndConvolve(resnet_out_channels, 128)
        self.conv2 = PadAndConvolve(128, 128)
        self.pool = nn.MaxPool2d(2)

        # Number of features input to the FC layer
        fc_size = (img_height // (2 ** 6)) * (img_width // (2 ** 6)) * 128
        self.fc = nn.Linear(fc_size, 2048)

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.resnet(x)[-1]
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = x.view(batchsize, -1)
        x = self.fc(x)
        return x.view(batchsize, 128, 4, 4)


class MonoLayoutDecoder(nn.Module):
    def __init__(
        self,
        in_channels: Optional[int] = 128,
        out_channels: Optional[int] = 2,
        layer_sizes: Optional[Iterable] = None,
    ):
        super(MonoLayoutDecoder, self).__init__()

        if not layer_sizes:
            layer_sizes = [256, 128, 64, 32, 16]
        layer_sizes.insert(0, in_channels)

        self.decoder_blocks = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.decoder_blocks.append(
                MonoLayoutDecoderBlock(layer_sizes[i], layer_sizes[i + 1])
            )

        self.final_conv = PadAndConvolve(layer_sizes[-1], out_channels)

    def forward(self, x, is_training=False):
        for i in range(len(self.decoder_blocks)):
            x = self.decoder_blocks[i](x)
        x = self.final_conv(x)
        if is_training:
            return x
        else:
            softmax = nn.Softmax2d()
            return softmax(x)


class MonoLayoutDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(MonoLayoutDiscriminator, self).__init__()

        layer_sizes = [64, 128, 256, 256, 512, 512]

        self.conv1 = nn.Conv2d(in_channels, layer_sizes[0], 3, 2, 1)
        self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv_blocks = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            inplanes = layer_sizes[i - 1]
            outplanes = layer_sizes[i]
            self.conv_blocks.append(MonoLayoutDiscriminatorBlock(inplanes, outplanes))

        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.leakyrelu1(self.conv1(x))
        for block in self.conv_blocks:
            x = block(x)
        batchsize = x.shape[0]
        x = x.view(x.shape[0], -1)
        return self.fc(x)


class MonoLayoutDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MonoLayoutDecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.functional.relu
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.bn2(self.conv2(x))


class MonoLayoutDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MonoLayoutDiscriminatorBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(x)))


class PadAndConvolve(nn.Module):
    def __init__(self, in_channels, out_channels, reflection=True):
        super(PadAndConvolve, self).__init__()
        self.pad = nn.ReflectionPad2d(1) if reflection else nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3)

    def forward(self, x):
        return self.conv(self.pad(x))


class ResNetFeatureExtractor(torchvision.models.ResNet):
    r"""We use a ResNet18 with a modified forward pass as our
    feature extractor.
    """

    def __init__(self, resnet_num_layers, pretrained=True):
        num_layers = resnet_num_layers
        block_sizes = [2, 2, 2, 2]

        super(ResNetFeatureExtractor, self).__init__(
            torchvision.models.resnet.BasicBlock, block_sizes
        )

        if pretrained:
            url = torchvision.models.resnet.model_urls["resnet18"]
            self.load_state_dict(torch.utils.model_zoo.load_url(url))
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = []
        x = (x - 0.45) / 0.225
        x = self.conv1(x)
        x = self.bn1(x)
        features.append(self.relu(x))
        features.append(self.layer1(self.maxpool(features[-1])))
        features.append(self.layer2(features[-1]))
        features.append(self.layer3(features[-1]))
        features.append(self.layer4(features[-1]))
        return features




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(2, 8, 3, 2, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(8, 16, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(16, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(32, 8, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(8, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
