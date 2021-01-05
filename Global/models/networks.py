# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from torch.nn.utils import spectral_norm

# from util.util import SwitchNorm2d
import torch.nn.functional as F

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "spectral":
        norm_layer = spectral_norm()
    elif norm_type == "SwitchNorm":
        norm_layer = SwitchNorm2d
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)


class GlobalGenerator_DCDCv2(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        k_size=3,
        n_downsampling=8,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
        opt=None,
    ):
        super(GlobalGenerator_DCDCv2, self).__init__()
        activation = nn.ReLU(True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, min(ngf, opt.mc), kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        ]
        ### downsample
        for i in range(opt.start_r):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    min(ngf * mult, opt.mc),
                    min(ngf * mult * 2, opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, opt.mc)),
                activation,
            ]
        for i in range(opt.start_r, n_downsampling - 1):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    min(ngf * mult, opt.mc),
                    min(ngf * mult * 2, opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, opt.mc)),
                activation,
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]
        mult = 2 ** (n_downsampling - 1)

        if opt.spatio_size == 32:
            model += [
                nn.Conv2d(
                    min(ngf * mult, opt.mc),
                    min(ngf * mult * 2, opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, opt.mc)),
                activation,
            ]
        if opt.spatio_size == 64:
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]
        model += [
            ResnetBlock(
                min(ngf * mult * 2, opt.mc),
                padding_type=padding_type,
                activation=activation,
                norm_layer=norm_layer,
                opt=opt,
            )
        ]
        # model += [nn.Conv2d(min(ngf * mult * 2, opt.mc), min(ngf, opt.mc), 1, 1)]
        if opt.feat_dim > 0:
            model += [nn.Conv2d(min(ngf * mult * 2, opt.mc), opt.feat_dim, 1, 1)]
        self.encoder = nn.Sequential(*model)

        # decode
        model = []
        if opt.feat_dim > 0:
            model += [nn.Conv2d(opt.feat_dim, min(ngf * mult * 2, opt.mc), 1, 1)]
        # model += [nn.Conv2d(min(ngf, opt.mc), min(ngf * mult * 2, opt.mc), 1, 1)]
        o_pad = 0 if k_size == 4 else 1
        mult = 2 ** n_downsampling
        model += [
            ResnetBlock(
                min(ngf * mult, opt.mc),
                padding_type=padding_type,
                activation=activation,
                norm_layer=norm_layer,
                opt=opt,
            )
        ]

        if opt.spatio_size == 32:
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, opt.mc),
                    min(int(ngf * mult / 2), opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), opt.mc)),
                activation,
            ]
        if opt.spatio_size == 64:
            model += [
                ResnetBlock(
                    min(ngf * mult, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]

        for i in range(1, n_downsampling - opt.start_r):
            mult = 2 ** (n_downsampling - i)
            model += [
                ResnetBlock(
                    min(ngf * mult, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, opt.mc),
                    min(int(ngf * mult / 2), opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), opt.mc)),
                activation,
            ]
        for i in range(n_downsampling - opt.start_r, n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, opt.mc),
                    min(int(ngf * mult / 2), opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), opt.mc)),
                activation,
            ]
        if opt.use_segmentation_model:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(min(ngf, opt.mc), output_nc, kernel_size=7, padding=0)]
        else:
            model += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(min(ngf, opt.mc), output_nc, kernel_size=7, padding=0),
                nn.Tanh(),
            ]
        self.decoder = nn.Sequential(*model)

    # def forward(self, input, flow="enc_dec"):
    #     if flow == "enc":
    #         return self.encoder(input)
    #     elif flow == "dec":
    #         return self.decoder(input)
    #     elif flow == "enc_dec":
    #         x = self.encoder(input)
    #         x = self.decoder(x)
    #         return x
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, input, flow="enc_dec"):
        if flow == "enc":
            h = self.encoder(input)
            if not self.eval_:
                mean = self.mean_layer(h)
                var = self.var_layer(h)
                h = self.reparameterize(mean, var)
                return h, mean, var
            else:
                return h
        elif flow == "dec":
            return self.decoder(input)
        elif flow == "enc_dec":
            z_x = self.encoder(input)
            if not self.eval_:
                mean = self.mean_layer(z_x)
                var = self.var_layer(z_x)
                z_x = self.reparameterize(mean, var)
                x = self.decoder(z_x)
                return z_x, mean, var, x
            return self.decoder(z_x)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(
        self, dim, padding_type, norm_layer, opt, activation=nn.ReLU(True), use_dropout=False, dilation=1
    ):
        super(ResnetBlock, self).__init__()
        self.opt = opt
        self.dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(self.dilation)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(self.dilation)]
        elif padding_type == "zero":
            p = self.dilation
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=self.dilation),
            norm_layer(dim),
            activation,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=1), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b : b + 1] == int(i)).nonzero()  # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[
                        indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]
                    ] = mean_feat
        return outputs_mean


def SN(module, mode=True):
    if mode:
        return torch.nn.utils.spectral_norm(module)

    return module


class NonLocalBlock2D_with_mask_Res(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels,
        mode="add",
        re_norm=False,
        temperature=1.0,
        use_self=False,
        cosin=False,
    ):
        super(NonLocalBlock2D_with_mask_Res, self).__init__()

        self.cosin = cosin
        self.renorm = re_norm
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.W = nn.Conv2d(
            in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0
        )
        # for pytorch 0.3.1
        # nn.init.constant(self.W.weight, 0)
        # nn.init.constant(self.W.bias, 0)
        # for pytorch 0.4.0
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.phi = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.mode = mode
        self.temperature = temperature
        self.use_self = use_self

        norm_layer = get_norm_layer(norm_type="instance")
        activation = nn.ReLU(True)

        model = []
        for i in range(3):
            model += [
                ResnetBlock(
                    inter_channels,
                    padding_type="reflect",
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=None,
                )
            ]
        self.res_block = nn.Sequential(*model)

    def forward(self, x, mask):  ## The shape of mask is Batch*1*H*W
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        if self.cosin:
            theta_x = F.normalize(theta_x, dim=2)
            phi_x = F.normalize(phi_x, dim=1)

        f = torch.matmul(theta_x, phi_x)

        f /= self.temperature

        f_div_C = F.softmax(f, dim=2)

        tmp = 1 - mask
        mask = F.interpolate(mask, (x.size(2), x.size(3)), mode="bilinear")
        mask[mask > 0] = 1.0
        mask = 1 - mask

        tmp = F.interpolate(tmp, (x.size(2), x.size(3)))
        mask *= tmp

        mask_expand = mask.view(batch_size, 1, -1)
        mask_expand = mask_expand.repeat(1, x.size(2) * x.size(3), 1)

        # mask = 1 - mask
        # mask=F.interpolate(mask,(x.size(2),x.size(3)))
        # mask_expand=mask.view(batch_size,1,-1)
        # mask_expand=mask_expand.repeat(1,x.size(2)*x.size(3),1)

        if self.use_self:
            mask_expand[:, range(x.size(2) * x.size(3)), range(x.size(2) * x.size(3))] = 1.0

        #    print(mask_expand.shape)
        #    print(f_div_C.shape)

        f_div_C = mask_expand * f_div_C
        if self.renorm:
            f_div_C = F.normalize(f_div_C, p=1, dim=2)

        ###########################

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        W_y = self.res_block(W_y)

        if self.mode == "combine":
            full_mask = mask.repeat(1, self.inter_channels, 1, 1)
            z = full_mask * x + (1 - full_mask) * W_y
        return z



class Z_xr_Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Z_xr_Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input, need_each_activation=False):
        each_activations = []
        if need_each_activation:
            x = input
            for i in range(len(self.model)):
                x = self.model[i](x)
                if isinstance(self.model[i], torch.nn.modules.activation.LeakyReLU):
                    each_activations.append(x)
        else:
            x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        # cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))
        # heatmap = torch.sum(x, dim=1, keepdim=True)
        x = self.pad(x)
        out = self.conv(x)
        if need_each_activation:
            return out, each_activations
        else:
            return out