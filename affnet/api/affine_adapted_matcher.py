import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

from PIL import Image
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F

from copy import deepcopy

from affnet.SparseImgRepresenter import ScaleSpaceAffinePatchExtractor

from affnet.architectures import AffNetFast, OriNetFast
from affnet.LAF import extract_patches,extract_patches_np_img,normalizeLAFs
from affnet.pytorch_sift import SIFTNet
import cv2
from lifetobot_sdk.Geometry import image_transformations




class AffineMatcher():
    def __init__(self, do_use_cuda=True, default_ha_tsh=-1):

        self.use_cuda = do_use_cuda
        self.th = default_ha_tsh
        self.AffNetPix = AffNetFast(PS=32)
        self.OriNetPix = OriNetFast(PS=32)
        self.SIFT = SIFTNet(patch_size=65, is_cuda=self.use_cuda)
        self.SIFT.eval()

        self.weightd_fname = '/media/Media/SWDEV/repos/MatchCoSeg/affnet/pretrained/AffNet.pth'
        self.orinet_weightd_fname = '/media/Media/SWDEV/repos/MatchCoSeg/affnet/pretrained/OriNet.pth'

        if not self.use_cuda:
            checkpoint = torch.load(self.weightd_fname, map_location=lambda storage, loc: storage)
            checkpoint_ori = torch.load(self.orinet_weightd_fname, map_location=lambda storage, loc: storage)

        else:
            checkpoint = torch.load(self.weightd_fname)
            checkpoint_ori = torch.load(self.orinet_weightd_fname)
        self.AffNetPix.load_state_dict(checkpoint['state_dict'])
        self.AffNetPix.eval()
        self.OriNetPix.load_state_dict(checkpoint_ori['state_dict'])
        self.OriNetPix.eval()

        self.HA = ScaleSpaceAffinePatchExtractor(mrSize=5.192, num_features=2000, border=5, num_Baum_iters=1, th=self.th,
                                            AffNet=self.AffNetPix, OriNet=self.OriNetPix)

        if self.use_cuda:
            self.HA = self.HA.cuda()

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.desc_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match_images(self,I1,I2):
        if I1.ndim == 3:
            I1 = np.mean(I1, axis=2)
        if I2.ndim == 3:
            I2 = np.mean(I2, axis=2)

        with torch.no_grad():
            var_image1 = torch.autograd.Variable(torch.from_numpy(I1.astype(np.float32)))
            var_image_reshape1 = var_image1.view(1, 1, var_image1.size(0), var_image1.size(1))
            var_image2 = torch.autograd.Variable(torch.from_numpy(I2.astype(np.float32)))
            var_image_reshape2 = var_image2.view(1, 1, var_image2.size(0), var_image2.size(1))
            if self.use_cuda:
                var_image_reshape1 = var_image_reshape1.cuda()
                var_image_reshape2 = var_image_reshape2.cuda()

            LAFs1, resp = self.HA(var_image_reshape1)
            sorted_idxs1 = resp.argsort(descending=True)
            LAFs2, resp2 = self.HA(var_image_reshape2)
            sorted_idxs2 = resp2.argsort(descending=True)
            torch_patches1 = extract_patches(var_image_reshape1,
                                             normalizeLAFs(LAFs1[sorted_idxs1, :, :], w=I1.shape[1], h=I1.shape[0]), PS=65,
                                             bs=32)
            desc1 = self.SIFT(torch_patches1)
            torch_patches2 = extract_patches(var_image_reshape2,
                                             normalizeLAFs(LAFs2[sorted_idxs2, :, :], w=I2.shape[1], h=I2.shape[0]),
                                             PS=65, bs=32)
            desc2 = self.SIFT(torch_patches2)

        matches = self.desc_matcher.knnMatch(desc1.cpu().numpy(), desc2.cpu().numpy(), k=2)
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
        idxs1 = [sorted_idxs1[match[0].queryIdx] for match in good_matches]
        idxs2 = [sorted_idxs2[match[0].trainIdx] for match in good_matches]
        dists = [match[0].distance for match in good_matches]

        affine_matches = dict(LAFs1=LAFs1[idxs1, :, :], LAFs2=LAFs2[idxs2, :, :], dists=dists)
        return affine_matches

if __name__ == '__main__':
    I1 = cv2.imread('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/img1.png')
    I2 = cv2.imread('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/img2.png')
    aff_matcher = AffineMatcher(do_use_cuda=True)
    out_dict = aff_matcher.match_images(I1,I2)
    pass