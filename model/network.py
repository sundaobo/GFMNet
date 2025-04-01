import torch
import torch.nn as nn
import torch.nn.functional as F
from vig_pytorch.pyramid_vig import pvig_s_224_gelu
from vig_pytorch.gcn_lib.torch_vertex import Grapher_group
from einops import rearrange
from loss.losses import Mutual_info_reg
from model.FreqFusion import FreqFusion

class Classifier(nn.Module):
    def __init__(self, in_chan=128, n_class=2):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(
                            nn.Conv2d(in_chan * 2, in_chan, kernel_size=3, padding=1, stride=1, bias=False),
                            nn.BatchNorm2d(in_chan),
                            nn.ReLU(),
                            nn.Conv2d(in_chan, n_class, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x = self.head(x)
        return x

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class GFMNet(nn.Module):
    def __init__(self,  backbone='resnet18', output_stride=16, img_size = 512, img_chan=3, chan_num = 32, n_class =2):
        super(GFMNet, self).__init__()
        BatchNorm = nn.BatchNorm2d
        self.backbone = pvig_s_224_gelu(pretrained=True)
        pretrain_path = "/mnt/ssd/sdb/Datasets/PGNET-FREQOur/pretrain/pvig_s_82.1.pth"
        state_dict = torch.load(pretrain_path)
        model_dict = self.backbone.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and "pos_embed" not in k}
        model_dict.update(filtered_dict)
        self.backbone.load_state_dict(model_dict, strict=False)
        print('Pretrain weights loaded.')
        
        self.decode32 = ConvBnLeakyRelu2d(chan_num, chan_num)
        self.decode16 = ConvBnLeakyRelu2d(chan_num, chan_num)
        self.decode8 = ConvBnLeakyRelu2d(chan_num, chan_num)
        self.decode4 = ConvBnLeakyRelu2d(chan_num, chan_num)

        self.ff11 = FreqFusion(chan_num, chan_num)
        self.ff12 = FreqFusion(chan_num, 2*chan_num)
        self.ff13 = FreqFusion(chan_num, 3*chan_num)
        
        self.ff21 = FreqFusion(chan_num, chan_num)
        self.ff22 = FreqFusion(chan_num, 2*chan_num)
        self.ff23 = FreqFusion(chan_num, 3*chan_num)

        self.classifier0 = Classifier(n_class = n_class)
        self.manloss4 = Mutual_info_reg(128, 256, 64)
        #

    def forward(self, img1, img2):
        # CNN backbone, feature extractor
        out1_s32, out1_s16, out1_s8, out1_s4 = self.backbone(img1) 
        # (8 32 16 16;)(8 32 32 32;)  (8 32 64 64;)  (8 32 128 128;)
        ##  x44: 32 16 16   x33: 32 32 32     x22: 32 64 64              x11: 32 128 128 
        out2_s32, out2_s16, out2_s8, out2_s4 = self.backbone(img2)
        
        out1_s32= self.decode32(out1_s32) # 8 32 16 16
        out2_s32 = self.decode32(out2_s32) # 8 32 16 16

        out1_s16= self.decode16(out1_s16) # 8 32 32 32
        out2_s16 = self.decode16(out2_s16) # 8 32 32 32

        out1_s8 = self.decode8(out1_s8)
        out2_s8 = self.decode8(out2_s8)

        out1_s4 = self.decode4(out1_s4)
        out2_s4 = self.decode4(out2_s4)
        
        _, out1_s16, x32_up = self.ff11(out1_s16, out1_s32)
        cat11 = torch.cat([out1_s16, x32_up], dim=1)
        _, out1_s8, x3216_up = self.ff12(out1_s8, cat11)
        cat12 = torch.cat([out1_s8, x3216_up], dim=1)
        _, out1_s4, x32168_up = self.ff13(out1_s4, cat12)
        x321684 = torch.cat([out1_s4, x32168_up], dim=1) # channel=4c, 1/4 img size

        _, out2_s16, y32_up = self.ff11(out2_s16, out2_s32)
        cat21 = torch.cat([out2_s16, y32_up], dim=1)
        _, out2_s8, y3216_up = self.ff12(out2_s8, cat21)
        cat22 = torch.cat([out2_s8, y3216_up], dim=1)
        _, out2_s4, y32168_up = self.ff13(out2_s4, cat22)
        y321684 = torch.cat([out2_s4, y32168_up], dim=1) # channel=4c, 1/4 img size                
        
        loss32 = self.manloss4(x321684, y321684)    
                            
        x16 = torch.cat([x321684, y321684], dim=1)                                          # 8 64  32  32 
        x16 = F.interpolate(x16, size=img1.shape[2:], mode='bicubic', align_corners=True) # 8 64 512 512
        x16 = self.classifier0(x16)                                                       # 8  2 512 512

        return x16, loss32
