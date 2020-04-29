import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from resnet_encoder import ResnetEncoder
# from .convlstm import ConvLSTM
from collections import OrderedDict



# Utils

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out



def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")





class Decoder(nn.Module):
    def __init__(self, num_ch_enc):
        super(Decoder, self).__init__()
        self.num_output_channels = 2
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.num_ch_concat = np.array([64, 128, 256, 512, 128])
        self.conv_mu = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv_log_sigma = nn.Conv2d(128, 128, 3, 1, 1)
        outputs = {}
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = 128 if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            num_ch_concat = self.num_ch_concat[i]
            self.convs[("upconv", i, 0)] = nn.Conv2d(num_ch_in, num_ch_out, 3, 1, 1) #Conv3x3(num_ch_in, num_ch_out)
            self.convs[("norm", i, 0)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("relu", i, 0)] =  nn.ReLU(True)

            # upconv_1
            self.convs[("upconv", i, 1)] = nn.Conv2d(num_ch_out, num_ch_out, 3, 1, 1) #ConvBlock(num_ch_out, num_ch_out)
            self.convs[("norm", i, 1)] = nn.BatchNorm2d(num_ch_out)

        self.convs["topview"] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        self.dropout = nn.Dropout3d(0.2)
        self.decoder = nn.ModuleList(list(self.convs.values()))



    def forward(self, x, is_training=True):
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = self.convs[("norm", i, 0)](x)
            x = self.convs[("relu", i, 0)](x)
            x = upsample(x)
            #x = torch.cat((x, features[i-6]), 1)
            x = self.convs[("upconv", i, 1)](x)
            x = self.convs[("norm", i, 1)](x)

        if is_training:
                x = self.convs["topview"](x) #self.softmax(self.convs["topview"](x))
        else:
                softmax = nn.Softmax2d()
                x = softmax(self.convs["topview"](x))
        #outputs["car"] = x
        return x #outputs







class Encoder(nn.Module):
    def __init__(self, num_layers, img_ht, img_wt, pretrained=True):
        super(Encoder, self).__init__()

        self.resnet_encoder = ResnetEncoder(num_layers, pretrained)#opt.weights_init == "pretrained"))
        num_ch_enc = self.resnet_encoder.num_ch_enc
        #convolution to reduce depth and size of features before fc
        self.conv1 = Conv3x3(num_ch_enc[-1], 128)
        self.conv2 = Conv3x3(128, 128)
        self.pool = nn.MaxPool2d(2)

        #fully connected
        curr_h = img_ht//(2**6)
        curr_w = img_wt//(2**6)
        features_in = curr_h*curr_w*128
        self.fc_mu = torch.nn.Linear(features_in, 2048)
        self.fc_sigma = torch.nn.Linear(features_in, 2048)
        self.fc = torch.nn.Linear(features_in, 2048)
        

    def forward(self, x, is_training= True):

        batch_size, c, h, w = x.shape
        x = self.resnet_encoder(x)[-1]
        x = self.pool(self.conv1(x))
        x = self.conv2(x)
        x = self.pool(x)
        return x



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



class Discriminator2(nn.Module):
    def __init__(self, input_ch):
        super(Discriminator, self).__init__()

        self.num_output_channels = 1
        self.num_ch_dec = np.array([64, 128, 256, 256, 512, 512])

        self.convs = OrderedDict()

        self.convs[("conv", 0)] = nn.Conv2d(input_ch, self.num_ch_dec[0], 3, 2, 1)
        self.convs[("lrelu", 0)] =  nn.LeakyReLU(0.2, True)

        for i in range(1, 6):
            num_ch_in = self.num_ch_dec[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("conv", i)] = nn.Conv2d(num_ch_in, num_ch_out, 3, 2, 1)
            self.convs[("norm", i)] = nn.BatchNorm2d(num_ch_out)
            self.convs[("lrelu", i)] =  nn.LeakyReLU(0.2, True)


        self.convs["linear"] = nn.Linear(2048, 1)
        self.encoder = nn.ModuleList(list(self.convs.values()))


    def forward(self, input_image):

        x = self.convs[("conv", 0)](input_image)
        x = self.convs[("lrelu", 0)](x)

        for i in range(1, 6):
            x = self.convs[("conv", i)](x)
            x = self.convs["norm", i](x)
            x = self.convs["lrelu", i](x)

        N, C, H, W = x.size()
        x = x.view(N, -1) 

        self.output = self.convs["linear"](x)

        return self.output    


