import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import cv2

from flowNet.flowClass import FlowNet


def central_crop(image_in, out_w, out_h):
    w, h = image_in.shape[0:2]
    x1 = int(round((w - out_w) / 2.))
    y1 = int(round((h - out_h) / 2.))
    if len(image_in.shape) == 3:
        return image_in[x1:x1 + out_w, y1:y1 + out_h, :]
    else:
        return image_in[x1:x1 + out_w, y1:y1 + out_h]

def parse_mat(M):
    dicts = []
    for k in range(M['patchPack'].shape[1]):
        data = M['patchPack'][0,k]
        dicts.append(dict(pSrc=data[0],
                          pDst=data[1],
                          lossFactor=data[2],
                          flowXX=data[7],
                          flowYY=data[8]))
    return dicts


do_conf_factor = True
flow_net = FlowNet(do_conf_factor=do_conf_factor)

import torch.optim as optim
from flowNet import losses
if do_conf_factor:
    critertion = losses.L1_factored(dict(C=2))
else:
    critertion = losses.L1Loss([])

optimizer = optim.SGD(flow_net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flow_net.to(device)


import scipy.io as spio
dicts = parse_mat(spio.loadmat('/home/erez/Downloads/rectifiedPatchesSIFT_128X128_withField_49.mat'))

batch_size = 128
valid_dim = 64
optimizer.zero_grad()
D = np.zeros((batch_size,6,134,134))
gt_flow_X_025 = np.zeros((batch_size,int(valid_dim/4),int(valid_dim/4)))
gt_flow_X_0125 = np.zeros((batch_size,int(valid_dim/8),int(valid_dim/8)))
gt_flow_Y_025 = np.zeros((batch_size,int(valid_dim/4),int(valid_dim/4)))
gt_flow_Y_0125 = np.zeros((batch_size,int(valid_dim/8),int(valid_dim/8)))
gt_mask_025 = np.zeros((batch_size,int(valid_dim/4),int(valid_dim/4)))
gt_mask_0125 = np.zeros((batch_size,int(valid_dim/8),int(valid_dim/8)))
factor_map_025 = np.zeros((batch_size,int(valid_dim/4),int(valid_dim/4)))
factor_map_0125 = np.zeros((batch_size,int(valid_dim/8),int(valid_dim/8)))
xx,yy = np.meshgrid(range(0,valid_dim),range(0,valid_dim))
#TODO: combing 'lossFactor' with valid mask
for k in range(0,batch_size):
    D[k,:,:,:] = np.expand_dims(np.vstack((cv2.resize(dicts[k]['pSrc'],(134,134)).transpose((2,0,1)),
                              cv2.resize(dicts[k]['pDst'],(134,134)).transpose((2,0,1)))), axis=0)
    valid_flowX = central_crop(dicts[k]['flowXX'], valid_dim, valid_dim)
    valid_flowY = central_crop(dicts[k]['flowYY'], valid_dim, valid_dim)
    gt_flow_X_025[k,:,:] = cv2.resize(valid_flowX/4, None, fx=0.25, fy=0.25)
    gt_flow_X_0125[k,:,:] = cv2.resize(valid_flowX/8, None, fx=0.125, fy=0.125)
    gt_flow_Y_025[k,:,:] = cv2.resize(valid_flowY/4, None, fx=0.25, fy=0.25)
    gt_flow_Y_0125[k,:,:] = cv2.resize(valid_flowY/8, None, fx=0.125, fy=0.125)
    if do_conf_factor:
        bad_x = np.logical_or((xx + valid_flowX) > valid_dim, (xx + valid_flowX) <  -1)
        bad_y = np.logical_or((yy + valid_flowY) > valid_dim, (yy + valid_flowY) < -1)
        valid_mask = np.logical_not(np.logical_or(bad_x,bad_y)).astype(np.float) # positive is valid
        gt_mask_025[k, :, :] = cv2.resize(valid_mask, None, fx=0.25, fy=0.25)
        gt_mask_0125[k, :, :] = cv2.resize(valid_mask, None, fx=0.125, fy=0.125)
        factor_map_025[k, :, :] = np.logical_or(gt_mask_025[k,:,:]==0,gt_mask_025[k,:,:]==1)
        factor_map_0125[k, :, :] = np.logical_or(gt_mask_0125[k, :, :] == 0, gt_mask_0125[k, :, :] == 1)
D = torch.Tensor((D-127)/128).to(device)



for k in range(200000):
    optimizer.zero_grad()
    flow_025, flow_0125 = flow_net(D)
    target_025 = np.concatenate((gt_flow_X_025[:,np.newaxis,:,:],
                                          gt_flow_Y_025[:, np.newaxis,:,:]),axis=1)
    target_0125 = np.concatenate((gt_flow_X_0125[:, np.newaxis, :, :],
                                 gt_flow_Y_0125[:, np.newaxis, :, :]), axis=1)
    if do_conf_factor:
        target_025 = np.concatenate((target_025,gt_mask_025[:, np.newaxis,:,:]),axis=1)
        target_0125 = np.concatenate((target_0125, gt_mask_0125[:, np.newaxis, :, :]), axis=1)
    loss_025 = critertion(flow_025, torch.Tensor(target_025).to(device),
                          torch.Tensor(factor_map_025).to(device))[0]
    loss_0125 = critertion(flow_0125,torch.Tensor(target_0125).to(device),
                           torch.Tensor(factor_map_0125).to(device))[0]
    loss = loss_025+loss_0125
    if not(k % 100):
        print(loss.item())
    loss.backward()

    # nn.utils.clip_grad_norm_(flow_net.parameters(), 1)
    optimizer.step()
pass
