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


flow_net = FlowNet()

import torch.optim as optim
from flowNet import losses
critertion = losses.L2Loss([])
optimizer = optim.SGD(flow_net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flow_net.to(device)


import scipy.io as spio
dicts = parse_mat(spio.loadmat('/media/rd/MyPassport/CoSegDataPasses/MPI_SINTEL_7Z/training/clean/alley_1/rectifiedPatchesSIFT_128X128_withField_49.mat'))

k = 0
optimizer.zero_grad()
D = np.expand_dims(np.vstack((cv2.resize(dicts[k]['pSrc'],(134,134)).transpose((2,0,1)),
                              cv2.resize(dicts[k]['pDst'],(134,134)).transpose((2,0,1)))), axis=0)
D = torch.Tensor(D).to(device)

valid_flowX = central_crop(dicts[k]['flowXX'], 64, 64)
valid_flowY = central_crop(dicts[k]['flowYY'], 64, 64)
gt_flow_X_025 = cv2.resize(valid_flowX, None, fx=0.25,fy=0.25)
gt_flow_X_0125 = cv2.resize(valid_flowX, None, fx=0.125,fy=0.125)
gt_flow_Y_025 = cv2.resize(valid_flowY, None, fx=0.25,fy=0.25)
gt_flow_Y_0125 = cv2.resize(valid_flowY, None, fx=0.125,fy=0.125)

for k in range(100):
    optimizer.zero_grad()
    flow_025, flow_0125 = flow_net(D)
    loss_025 = critertion(flow_025,
                          torch.Tensor(np.concatenate((gt_flow_X_025[np.newaxis,np.newaxis,:,:],
                                          gt_flow_Y_025[np.newaxis, np.newaxis,:,:]),axis=1)).to(device))

    loss_x025 = critertion(flow_025[:, 0, :,:], torch.Tensor(np.expand_dims(gt_flow_X_025, axis=0)).to(device))
    loss_x0125 = critertion(flow_0125[:, 0, :,:], torch.Tensor(np.expand_dims(gt_flow_X_0125, axis=0)).to(device))
    loss_y025 = critertion(flow_025[:, 1, :,:], torch.Tensor(np.expand_dims(gt_flow_Y_025, axis=0)).to(device))
    loss_x0125 = critertion(flow_0125[:, 1, :,:], torch.Tensor(np.expand_dims(gt_flow_Y_0125, axis=0)).to(device))
    loss = loss_x025 + loss_x0125 + loss_y025 + loss_x0125
    print(loss.item())
    loss.backward()
    optimizer.step()


