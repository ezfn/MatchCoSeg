import torch
import torchvision
import torch.nn as nn
import numpy as np
from flowNet.Utils import upsize_linear

from flowNet.flowClass import FlowNet
import torch.optim as optim
from flowNet import losses
import os
import glob
from Visualization.torch_vis import imshow_tensor, plot_3ch_kernels, multi_imshow_tensor, \
    multi_flow_show_tensor, show_3ch_pairs
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
        self.upsize_4 = upsize_linear(scale_factor=4)
        self.optimizer = optim.Adam(self.flow_net.parameters(), lr=1e-5, weight_decay=0.001)
        # self.optimizer = optim.SGD(self.flow_net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.flow_net.to(device)

    def train_sample(self, sample, summary_writer=None, ctr=0):
        self.optimizer.zero_grad()
        flow_025, flow_0125 = self.flow_net(sample['patches'])
        loss_025, _, per_example_errs025 = self.critertion(flow_025, sample['gt_flow_025'],
                                                           sample['factor_map_025'])
        loss_0125 = self.critertion(flow_0125, sample['gt_flow_0125'],
                                                            sample['factor_map_0125'])[0]
        loss = loss_025/2 + loss_0125/2
        # loss = loss_0125
        if summary_writer is not None:
            src_images = self.flow_net.central_crop(sample['patches'][:, 0:3, :, :],64,64)
            tgt_images = self.flow_net.central_crop(sample['patches'][:, 3:6, :, :],64,64)
            flow_025_cpy = flow_025.clone()
            full_flow_gt = self.upsize_4(sample['gt_flow_025'])
            full_flow_gt[:,0:2,:,:] = 4*full_flow_gt[:,0:2,:,:]
            flow_025_cpy[:, 2, :, :] = torch.sigmoid(flow_025_cpy[:, 2, :, :])
            full_flow_pred = self.upsize_4(flow_025_cpy)
            full_flow_pred[:, 0:2, :, :] = 4 * full_flow_pred[:, 0:2, :, :]
            warped_images_gt = self.flow_net.warp(tgt_images, full_flow_gt)
            gt_mask = (full_flow_gt[:,None,2,:,:].repeat(1,3,1,1) > 0).type('torch.cuda.FloatTensor')
            warped_images_gt *= gt_mask
            warped_images_pred = self.flow_net.warp(tgt_images, full_flow_pred)
            warped_images_pred *= full_flow_pred[:,None,2,:,:].repeat(1,3,1,1)
            src_images *= gt_mask
            tgt_images = 0.5 + (tgt_images.transpose(1, 2).transpose(2, 3) * 0.5)
            idxs = [0, 1, 2, 3]
            # fig_tgts = plot_3ch_kernels(0.5 + (tgt_images[idxs].transpose(1, 2).transpose(2, 3) * 0.5))
            # fig_tgts.suptitle('original_tgts')
            # summary_writer.add_figure('vis/tgts', fig_tgts, ctr)
            fig_gtw = show_3ch_pairs(0.5 + (warped_images_gt[idxs].transpose(1, 2).transpose(2, 3) * 0.5),
                                     tgt_images[idxs], num_cols=1)
            fig_gtw.suptitle('ground_truth_warped')
            fig_predw = show_3ch_pairs(0.5 + (warped_images_pred[idxs].transpose(1, 2).transpose(2, 3) * 0.5),
                                       tgt_images[idxs], num_cols=1)
            fig_predw.suptitle('pred_warped. Loss:{}'.format(loss))
            fig_src_masked = show_3ch_pairs(0.5 + (src_images[idxs].transpose(1, 2).transpose(2, 3) * 0.5),
                                            tgt_images[idxs], num_cols=1)
            fig_src_masked.suptitle('masked_src_patches')
            fig_pred_flow = multi_flow_show_tensor(flow_025_cpy[idxs], flow_losses=per_example_errs025.detach().cpu().numpy())
            fig_pred_flow.suptitle('pred_flow')
            fig_gt_flow = multi_flow_show_tensor(sample['gt_flow_025'][idxs])
            fig_gt_flow.suptitle('gt_flow')
            summary_writer.add_figure('vis/gtw',fig_gtw, ctr)
            summary_writer.add_figure('vis/predw', fig_predw, ctr)
            summary_writer.add_figure('vis/src_masked', fig_src_masked, ctr)
            summary_writer.add_figure('vis/pred_flow', fig_pred_flow, ctr)
            summary_writer.add_figure('vis/gt_flow', fig_gt_flow, ctr)

        if False:
            from lifetobot_sdk.Visualization import drawers as d
            # from affnet.dataset_passes.flow_vis import flow_compute_color
            from affnet.dataset_passes.flowlib import compute_color
            idx = 6
            mask_est = torch.sigmoid(flow_0125[idx,2,:,:]).cpu().detach().numpy() > 0.5
            u_est = mask_est * flow_0125[idx,0,:,:].cpu().detach().numpy()
            v_est = mask_est * flow_0125[idx,1,:,:].cpu().detach().numpy()
            C = compute_color(u_est, v_est)
            d.imshow(C.astype(np.uint8),0,'est')
            mask = sample['gt_flow_0125'][idx, 2, :, :].cpu().detach().numpy() == 1
            u = mask * sample['gt_flow_0125'][idx, 0, :, :].cpu().detach().numpy()
            v = mask * sample['gt_flow_0125'][idx,1,:,:].cpu().detach().numpy()
            C_gt = compute_color(u, v)
            d.imshow(C_gt.astype(np.uint8), 0, 'gt')
            d.imshow((127+128*sample['patches'][idx, 0:3, :, :].cpu().detach().numpy().transpose(1, 2, 0)).astype(np.uint8), 0, 'src_img')
            d.imshow((127+128*sample['patches'][idx, 3:6, :, :].cpu().detach().numpy().transpose(1, 2, 0)).astype(np.uint8), 0, 'tgt_img')

        # # if not (k % 100):
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

