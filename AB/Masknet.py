"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
os.environ['PYTHON_EGG_CACHE'] = 'tmp/'  # a writable directory
import sys
from correlation import Correlation
# from correlation_package.modules.corr import Correlation
import numpy as np

# __all__ = [
#     'pwc_dc_net'
# ]


class FlowNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """

    def __init__(self, ):
        """
        input: corr_d --- maximum displacement (for correlation. default: 4), after warping

        """
        super(FlowNet, self).__init__()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
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
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
        # grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            xx = xx.cuda()
            yy = yy.cuda()
            # grid = grid.cuda()
        xx = Variable(xx) + flo[:, 0, None, :, :]
        yy = Variable(yy) + flo[:, 1, None, :, :]
        # vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        # vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        # vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        xx = 2.0 * xx / max(W - 1, 1) - 1.0
        yy = 2.0 * yy / max(H - 1, 1) - 1.0
        vgrid = torch.cat((xx, yy), 1).permute(0, 2, 3, 1)

        # effectively mask out pixels close to warping boundaries
        # TODO: maybe put here our estimated mask
        # vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def central_crop(self, tensor_in, out_w, out_h):
        w, h = tensor_in.shape[2:4]
        x1 = int(round((w - out_w) / 2.))
        y1 = int(round((h - out_h) / 2.))
        return tensor_in[:, :, x1:x1 + out_w, y1:y1 + out_h]

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
        corr3 = self.corr(c13, c23)
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
        up_feat3 = self.upfeat3(x) # two/three output channels!

        # warp according to the estimated flow
        warp2 = self.warp(self.central_crop(c22, up_flow3.shape[2], up_flow3.shape[3]), up_flow3)
        c12_cropped = self.central_crop(c12, warp2.shape[2], warp2.shape[3])
        corr2 = self.corr(c12_cropped, warp2)
        corr2 = self.leakyRELU(corr2)
        #TODO: consider also multiplying corr2 with the estimated mask at flow3[:,2,:,:]
        x = torch.cat((corr2, c12_cropped, up_flow3, up_feat3), 1)
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

# def pwc_dc_net(path=None):
#     model = FlowNet()
#     if path is not None:
#         data = torch.load(path)
#         if 'state_dict' in data.keys():
#             model.load_state_dict(data['state_dict'])
#         else:
#             model.load_state_dict(data)
#     return model
