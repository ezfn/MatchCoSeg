import torch
import torchvision
import torch.nn as nn
import numpy as np

from flowNet.flowClass import FlowNet
import torch.optim as optim
from flowNet import losses
import os
import glob
class Trainer(object):

    def __init__(self, train_dir = None, do_conf_factor = True):
        self.prev_loss = 1000
        self.train_dir = train_dir
        model_files = glob.glob(os.path.join(self.train_dir,'*.pt'))
        if not len(model_files):
            self.flow_net = FlowNet(do_conf_factor=do_conf_factor)
        else:
            model_files.sort(key=os.path.getmtime)
            self.load_model(model_files[-1])
        if do_conf_factor:
            self.critertion = losses.L1_factored(dict(C=0.2))
        else:
            self.critertion = losses.L1Loss([])
        self.optimizer = optim.SGD(self.flow_net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.flow_net.to(device)

    def train_sample(self, sample):
        self.optimizer.zero_grad()
        flow_025, flow_0125 = self.flow_net(sample['patches'])
        loss_025 = self.critertion(flow_025, sample['gt_flow_025'], sample['factor_map_025'])[0]
        loss_0125 = self.critertion(flow_0125, sample['gt_flow_0125'], sample['factor_map_0125'])[0]
        loss = loss_025 + loss_0125
        # from lifetobot_sdk.Visualization import drawers as d
        # # from affnet.dataset_passes.flow_vis import flow_compute_color
        # from affnet.dataset_passes.flowlib import compute_color
        # idx = 40
        # mask_est = torch.sigmoid(flow_025[idx,2,:,:]).cpu().detach().numpy() > 0.5
        # u_est = mask_est * flow_025[idx,0,:,:].cpu().detach().numpy()
        # v_est = mask_est * flow_025[idx,1,:,:].cpu().detach().numpy()
        # C = compute_color(u_est, v_est)
        # d.imshow(C.astype(np.uint8),0,'est')
        # mask = sample['gt_flow_025'][idx, 2, :, :].cpu().detach().numpy() == 1
        # u = mask * sample['gt_flow_025'][idx, 0, :, :].cpu().detach().numpy()
        # v = mask * sample['gt_flow_025'][idx,1,:,:].cpu().detach().numpy()
        # C_gt = compute_color(u, v)
        # d.imshow(C_gt.astype(np.uint8), 0, 'gt')
        # if not (k % 100):
        #     print(loss.item())
        self.prev_loss = loss.item()
        loss.backward()

        # nn.utils.clip_grad_norm_(flow_net.parameters(), 1)
        self.optimizer.step()
        return loss

    def load_model(self, model_file_name):
        self.flow_net = torch.load(model_file_name)

    def save_model(self, step=0):
        torch.save(self.flow_net, os.path.join(self.train_dir, 'model_{}.pt'.format(step)))

