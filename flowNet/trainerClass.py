import torch
import torchvision
import torch.nn as nn
import numpy as np

from flowNet.flowClass import FlowNet
import torch.optim as optim
from flowNet import losses

class Trainer(object):

    def __init__(self, train_dir = None, do_conf_factor = True):
        self.flow_net = FlowNet(do_conf_factor=do_conf_factor)
        if do_conf_factor:
            self.critertion = losses.L1_factored(dict(C=2))
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
        # if not (k % 100):
        #     print(loss.item())
        loss.backward()

        # nn.utils.clip_grad_norm_(flow_net.parameters(), 1)
        self.optimizer.step()
        return loss

    def save_weights(self):
        pass

