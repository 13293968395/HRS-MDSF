from torch import nn
import model2
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from model2 import Encoder, Decoder


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SFconv(nn.Module):
    def __init__(self, features, mode, M=2, r=2, L=32) -> None:
        super().__init__()
        
        d = max(int(features/r), L)
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(features, features, 1, 1, 0)

        if mode[0] == 'train':
            self.gap = nn.AdaptiveAvgPool2d(1)
        

    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)
        
        attention_vectors = torch.cat([high_att, low_att], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)

        fea_high = high * high_att
        fea_low = low * low_att
        
        out = self.out(fea_high + fea_low) 
        return out

# MDSF
class dynamic_filter(nn.Module):
    def __init__(self, inchannels, mode, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.modulate = SFconv(inchannels, mode)

    def forward(self, x):
        identity_input = x 
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)
    
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        out = self.modulate(low_part, out_high)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mode, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.filter = filter

        self.dyna = dynamic_filter(in_channel//2, mode) if filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel//2, mode, kernel_size=5) if filter else nn.Identity()

        self.localap = Patch_ap(mode, in_channel//2, patch_size=2)
        self.global_ap = Gap(in_channel//2, mode)


    def forward(self, x):
        out = self.conv1(x)
       
       # MDSF
        if self.filter:
            k3, k5 = torch.chunk(out, 2, dim=1)
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            out = torch.cat((out_k3, out_k5), dim=1)
        
        # MCSF
        non_local, local = torch.chunk(out, 2, dim=1)
        non_local = self.global_ap(non_local)
        local = self.localap(local) 
        out = torch.cat((non_local, local), dim=1)
        out = self.conv2(out)
        return out + x

import torch.nn as nn
import model2

class DMPHN(nn.Module):
    def __init__(self):
        super(DMPHN, self).__init__()
        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()
        for s in ['s1', 's2', 's3', 's4']:
            self.encoder[s] = nn.ModuleDict()
            self.decoder[s] = nn.ModuleDict()
            for lv in ['lv1', 'lv2', 'lv3']:
                self.encoder[s][lv] = model2.Encoder()
                self.decoder[s][lv] = model2.Decoder()

    def forward(self, inputs):

        images = {}
        feature = {}
        residual = {}
        for s in ['s1', 's2', 's3', 's4']:
            feature[s] = {} # encoder output
            residual[s] = {} # decoder output

        images['lv1'] = Variable(inputs - 0.5).cuda()
        H = images['lv1'].size(2)
        W = images['lv1'].size(3)

        images['lv2_1'] = images['lv1'][:, :, 0:int(H / 2), :]
        images['lv2_2'] = images['lv1'][:, :, int(H / 2):H, :]
        images['lv3_1'] = images['lv2_1'][:, :, :, 0:int(W / 2)]
        images['lv3_2'] = images['lv2_1'][:, :, :, int(W / 2):W]
        images['lv3_3'] = images['lv2_2'][:, :, :, 0:int(W / 2)]
        images['lv3_4'] = images['lv2_2'][:, :, :, int(W / 2):W]

        for s, ps in zip(['s1', 's2', 's3', 's4'], [None, 's1', 's2', 's3']):
            if ps is None:
                feature[s]['lv3_1'] = self.encoder[s]['lv3'](images['lv3_1'])
                feature[s]['lv3_2'] = self.encoder[s]['lv3'](images['lv3_2'])
                feature[s]['lv3_3'] = self.encoder[s]['lv3'](images['lv3_3'])
                feature[s]['lv3_4'] = self.encoder[s]['lv3'](images['lv3_4'])
            else:
                feature[s]['lv3_1'] = self.encoder[s]['lv3'](images['lv3_1'] + residual[ps]['lv1'][:, :, 0:int(H / 2), 0:int(W / 2)])
                feature[s]['lv3_2'] = self.encoder[s]['lv3'](images['lv3_2'] + residual[ps]['lv1'][:, :, 0:int(H / 2), int(W / 2):W])
                feature[s]['lv3_3'] = self.encoder[s]['lv3'](images['lv3_3'] + residual[ps]['lv1'][:, :, int(H / 2):H, 0:int(W / 2)])
                feature[s]['lv3_4'] = self.encoder[s]['lv3'](images['lv3_4'] + residual[ps]['lv1'][:, :, int(H / 2):H, int(W / 2):W])

            feature[s]['lv3_top'] = torch.cat((feature[s]['lv3_1'], feature[s]['lv3_2']), 3)
            feature[s]['lv3_bot'] = torch.cat((feature[s]['lv3_3'], feature[s]['lv3_4']), 3)

            if ps is not None:
                feature[s]['lv3_top'] += feature[ps]['lv3_top']
                feature[s]['lv3_bot'] += feature[ps]['lv3_bot']

            residual[s]['lv3_top'] = self.decoder[s]['lv3'](feature[s]['lv3_top'])
            residual[s]['lv3_bot'] = self.decoder[s]['lv3'](feature[s]['lv3_bot'])

            ########################################

            if ps is None:
                feature[s]['lv2_1'] = self.encoder[s]['lv2'](images['lv2_1'] + residual[s]['lv3_top'])
                feature[s]['lv2_2'] = self.encoder[s]['lv2'](images['lv2_2'] + residual[s]['lv3_bot'])
            else:
                feature[s]['lv2_1'] = self.encoder[s]['lv2'](images['lv2_1'] + residual[s]['lv3_top'] + residual[ps]['lv1'][:, :, 0:int(H / 2), :])
                feature[s]['lv2_2'] = self.encoder[s]['lv2'](images['lv2_2'] + residual[s]['lv3_bot'] + residual[ps]['lv1'][:, :, int(H / 2):H, :])

            feature[s]['lv2_1'] += feature[s]['lv3_top']
            feature[s]['lv2_2'] += feature[s]['lv3_bot']

            if ps is not None:
                feature[s]['lv2_1'] += feature[ps]['lv2_1']
                feature[s]['lv2_2'] += feature[ps]['lv2_2']

            feature[s]['lv2'] = torch.cat((feature[s]['lv2_1'], feature[s]['lv2_2']), 2)

            residual[s]['lv2'] = self.decoder[s]['lv2'](feature[s]['lv2'])

            ########################################
            # if ps is not None:
            #     residual[s]['lv2'] += residual[ps]['lv2']
            if ps is None:
                feature[s]['lv1'] = self.encoder[s]['lv1'](images['lv1'] + residual[s]['lv2'])
            else:
                feature[s]['lv1'] = self.encoder[s]['lv1'](images['lv1'] + residual[s]['lv2'] + residual[ps]['lv1'])# + feature[s]['lv2']

            feature[s]['lv1'] += feature[s]['lv2']

            if ps is not None:
                feature[s]['lv1'] += feature[ps]['lv1']

            residual[s]['lv1'] = self.decoder[s]['lv1'](feature[s]['lv1'])


        return (residual['s1']['lv1'], residual['s2']['lv1'],  residual['s3']['lv1'], residual['s4']['lv1'])
    
