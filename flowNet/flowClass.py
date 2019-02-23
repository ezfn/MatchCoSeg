"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from Utils import conv,predict_flow,deconv,context_network
os.environ['PYTHON_EGG_CACHE'] = 'tmp/'  # a writable directory
import sys
from correlation_package.modules.corr import Correlation
import numpy as np

# __all__ = [
#     'pwc_dc_net'
# ]


class FlowNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """

    def __init__(self, corr_d=4):
        """
        input: corr_d --- maximum displacement (for correlation. default: 4), after warping

        """
        super(FlowNet, self).__init__()

        self.corr_d = corr_d

        self.lvl_in_ch = [3,16,32,64,96,128,196]
        self.combined_conv_ch = [128, 128, 96, 64, 32] # this is completely arbitrary and configurable!
        self.combined_conv_acc_ch = np.cumsum(self.combined_conv_ch)
        n_corr_dims = (2 * self.corr_d + 1) ** 2 # num of channels produced by correlation (== matched filter scan area)

        self.conv1a = conv(self.lvl_in_ch[0], self.lvl_in_ch[1], kernel_size=3, stride=2)
        self.conv1aa = conv(self.lvl_in_ch[1], self.lvl_in_ch[1], kernel_size=3, stride=1)
        self.conv1b = conv(self.lvl_in_ch[1], self.lvl_in_ch[1], kernel_size=3, stride=1)
        self.conv2a = conv(self.lvl_in_ch[1], self.lvl_in_ch[2], kernel_size=3, stride=2)
        self.conv2aa = conv(self.lvl_in_ch[2], self.lvl_in_ch[2], kernel_size=3, stride=1)
        self.conv2b = conv(self.lvl_in_ch[2], self.lvl_in_ch[2], kernel_size=3, stride=1)
        self.conv3a = conv(self.lvl_in_ch[2], self.lvl_in_ch[3], kernel_size=3, stride=2)
        self.conv3aa = conv(self.lvl_in_ch[3], self.lvl_in_ch[3], kernel_size=3, stride=1)
        self.conv3b = conv(self.lvl_in_ch[3], self.lvl_in_ch[3], kernel_size=3, stride=1)

        self.corr = Correlation(pad_size=self.corr_d, kernel_size=1, max_displacement=self.corr_d, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)


        in_dims = n_corr_dims
        self.conv3_0 = conv(in_dims, self.combined_conv_ch[0], kernel_size=3, stride=1)
        self.conv3_1 = conv(in_dims + self.combined_conv_acc_ch[0], self.combined_conv_ch[1], kernel_size=3, stride=1)
        self.conv3_2 = conv(in_dims + self.combined_conv_acc_ch[1], self.combined_conv_ch[2], kernel_size=3, stride=1)
        self.conv3_3 = conv(in_dims + self.combined_conv_acc_ch[2], self.combined_conv_ch[3], kernel_size=3, stride=1)
        self.conv3_4 = conv(in_dims + self.combined_conv_acc_ch[3], self.combined_conv_ch[4], kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(in_dims + self.combined_conv_acc_ch[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(in_dims + self.combined_conv_acc_ch[4], 2, kernel_size=4, stride=2, padding=1)

        in_dims = n_corr_dims + self.lvl_in_ch[2] + 2 + 2 # concatenation of {corr, src features, prev flow and prev combined features}
        self.conv2_0 = conv(in_dims, self.combined_conv_ch[0], kernel_size=3, stride=1)
        self.conv2_1 = conv(in_dims + self.combined_conv_acc_ch[0], self.combined_conv_ch[1], kernel_size=3, stride=1)
        self.conv2_2 = conv(in_dims + self.combined_conv_acc_ch[1], self.combined_conv_ch[2], kernel_size=3, stride=1)
        self.conv2_3 = conv(in_dims + self.combined_conv_acc_ch[2], self.combined_conv_ch[3], kernel_size=3, stride=1)
        self.conv2_4 = conv(in_dims + self.combined_conv_acc_ch[3], self.combined_conv_ch[4], kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(in_dims + self.combined_conv_acc_ch[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.context_net, out_planes = context_network(in_dims + self.combined_conv_acc_ch[4])
        self.flow_from_context = predict_flow(out_planes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        # effectively mask out pixels close to warping boundaries
        # TODO: maybe put here our estimated mask
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def forward(self, x):
        im1 = x[:, :3, :, :]
        im2 = x[:, 3:, :, :]

        # process the two images in a siamese fashion (each pyramid level has it's own weights!)
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))

        # correlate the two channels and process the result
        corr3 = self.corr(c13,c23)
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((self.conv3_0(corr3), corr3), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        # calculate the flow (each pyramid level has it's own weights!)
        flow3 = self.predict_flow3(x)
        # upsample the flow (each pyramid level has it's own weights!)
        up_flow3 = self.deconv3(flow3)
        # upsample the processed corr (each pyramid level has it's own weights!)
        up_feat3 = self.upfeat3(x) # two output channels!

        # warp according to the estimated flow
        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)

        # predict flow using the 'context' network and add it to the standard estimation
        # this is done only in the final stage
        x = self.context_net(x)
        flow2 += self.flow_from_context(x)

        if self.training:
            return flow2, flow3
        else:
            return flow2

def pwc_dc_net(path=None):
    model = FlowNet()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model
