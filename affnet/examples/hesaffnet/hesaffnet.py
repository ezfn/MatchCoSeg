#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
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
from affnet.LAF import denormalizeLAFs, LAFs2ell, abc2A
from affnet.Utils import line_prepender
from affnet.architectures import AffNetFast, OriNetFast
from affnet.LAF import extract_patches,extract_patches_np_img,normalizeLAFs
from affnet.pytorch_sift import SIFTNet
import cv2
from lifetobot_sdk.Geometry import image_transformations


USE_CUDA = False
th = 28.41 # default threshold for HessianAffine 
th = -1
try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
    nfeats = int(sys.argv[3])
except:
    input_img_fname = 'img/cat.png'
    output_fname = 'bla.txt'
    nfeats = 2000
    # print("Wrong input format. Try python hesaffnet.py imgs/cat.png cat.txt 2000")
    # sys.exit(1)

img = Image.open(input_img_fname).convert('RGB')
img_np = np.array(img)
Ha = np.eye(3);Ha[0,0] = 1.1;Ha[1,1] = 1.4;Ha[0,2] = 0.3
img2_np= image_transformations.homogenous_transform_image(img_np,Ha,img_np.shape)
img2 = np.mean(img2_np, axis = 2)
img = np.mean(img_np, axis = 2)
# img = np.hstack((img,img))
# img = cv2.resize(img,None,fx=2.0,fy=1.0)
AffNetPix = AffNetFast(PS = 32)
OriNetPix = OriNetFast(PS = 32)
weightd_fname = '../../pretrained/AffNet.pth'

if not USE_CUDA:
    checkpoint = torch.load(weightd_fname, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(weightd_fname)
SIFT = SIFTNet(patch_size = 65, is_cuda=USE_CUDA)
SIFT.eval()
AffNetPix.load_state_dict(checkpoint['state_dict'])
AffNetPix.eval()
HA = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = nfeats, border = 5, num_Baum_iters = 1, th = th,
                                     AffNet = AffNetPix, OriNet= OriNetPix)
if USE_CUDA:
    HA = HA.cuda()

# imgs = np.concatenate((img[:,:,None],img[:,:,None]),axis=2)
with torch.no_grad():
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)))
    var_image_reshape = var_image.view(1, 1, var_image.size(0), var_image.size(1))
    var_image2 = torch.autograd.Variable(torch.from_numpy(img2.astype(np.float32)))
    var_image_reshape2 = var_image2.view(1, 1, var_image2.size(0), var_image2.size(1))


if USE_CUDA:
    var_image_reshape = var_image_reshape.cuda()
    var_image_reshape2 = var_image_reshape2.cuda()



FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
# bf = cv2.BFMatcher(crossCheck=True)
for k in range(20):
    t0 = time.time()

    with torch.no_grad():
        LAFs, resp = HA(var_image_reshape)
        best_idxs = resp.argsort(descending=True)
        LAFs2, resp2 = HA(var_image_reshape2)
        best_idxs2 = resp2.argsort(descending=True)
    # ells = LAFs2ell(LAFs.data.cpu().numpy())

    # np.savetxt(output_fname, ells, delimiter=' ', fmt='%10.10f')
    # line_prepender(output_fname, str(len(ells)))
    # line_prepender(output_fname, '1.0')
    #     aa = extract_patches_np_img(var_image_reshape, normalizeLAFs(LAFs[best_idxs,:,:]), PS = 65, bs = 32)
        torch_patches = extract_patches(var_image_reshape, normalizeLAFs(LAFs[best_idxs,:,:],w=img.shape[1],h=img.shape[0]), PS = 65, bs = 32)
        res = SIFT(torch_patches)
        torch_patches2 = extract_patches(var_image_reshape2, normalizeLAFs(LAFs2[best_idxs2,:,:],w=img2.shape[1],h=img2.shape[0]), PS = 65, bs = 32)
        res2 = SIFT(torch_patches2)

    single_extraction_time = time.time()-t0
    print('extraction time:', single_extraction_time)
    # Find point matches
    t0 = time.time()
    des1 = res.cpu().numpy()
    des2 = res2.cpu().numpy()
    matches = flann.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
    
    print('total time:', single_extraction_time + time.time()-t0)

    pass
