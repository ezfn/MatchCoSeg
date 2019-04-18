
from torch.utils.data import dataset
from torchvision import transforms
import scipy.io as spio
import os
from pathlib import Path
import numpy as np
import cv2
import torch

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

class ToTensor(object):
    """Convert patches in sample to Tensors."""
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, sample):

        sample['patches'] = torch.Tensor(sample['patches']).to(self.device)
        sample['gt_flow_025'] = torch.Tensor(sample['gt_flow_025']).to(self.device)
        sample['gt_flow_0125'] = torch.Tensor(sample['gt_flow_0125']).to(self.device)
        sample['factor_map_025'] = torch.Tensor(sample['factor_map_025']).to(self.device)
        sample['factor_map_0125'] = torch.Tensor(sample['factor_map_0125']).to(self.device)
        return sample



class PickleBasedCosegDataset(dataset.Dataset):
    def __init__(self, root_dir, example_format, do_conf_factor=True, transform_list = [ToTensor()]):
        self.example_format = example_format
        self.root_dir = Path(root_dir)
        self.example_paths = self.root_dir.glob(example_format)
        self.example_paths = [path.as_posix() for path in self.example_paths]

        self.do_conf_factor = do_conf_factor
        self.transform = transforms.Compose(transform_list)
        self.last_sample = None

    def __len__(self):
        return len(self.example_paths)

    def __getitem__(self, idx):
        input_file = self.example_paths[idx]
        import gzip
        import pickle as pkl
        try:
            with gzip.open(input_file,'rb') as f:
                examples = pkl.load(f)
        except:
            print('########frame examples - bad file!##########')
            return self.last_sample
        examples = [d for d in examples if d['src_patch'].size > 0]
        batch_size = len(examples)
        if batch_size == 0:
            print('########frame examples are empty, giving last sample!##########')
            return self.last_sample
        # batch_size = min(256, len(examples))
        valid_dim = 64
        D = np.zeros((batch_size, 6, 134, 134))
        if self.do_conf_factor:
            gt_flow_025 = np.zeros((batch_size, 3, int(valid_dim / 4), int(valid_dim / 4)))
            gt_flow_0125 = np.zeros((batch_size, 3, int(valid_dim / 8), int(valid_dim / 8)))
        else:
            gt_flow_025 = np.zeros((batch_size, 2, int(valid_dim / 4), int(valid_dim / 4)))
            gt_flow_0125 = np.zeros((batch_size, 2, int(valid_dim / 8), int(valid_dim / 8)))

        factor_map_025 = np.zeros((batch_size, int(valid_dim / 4), int(valid_dim / 4)))
        factor_map_0125 = np.zeros((batch_size, int(valid_dim / 8), int(valid_dim / 8)))
        xx, yy = np.meshgrid(range(0, valid_dim), range(0, valid_dim))
        xx025, yy025 = np.meshgrid(range(0, int(valid_dim / 4)), range(0, int(valid_dim / 4)))
        xx0125, yy0125 = np.meshgrid(range(0, int(valid_dim / 8)), range(0, int(valid_dim / 8)))

        # TODO: combing 'lossFactor' with valid mask
        for k in range(0, batch_size):
            D[k, :, :, :] = np.expand_dims(np.vstack((examples[k]['src_patch'].transpose((2, 0, 1)),
                                                      examples[k]['tgt_patch'].transpose((2, 0, 1)))),
                                           axis=0)
            valid_flowX = central_crop(examples[k]['flowXX'], valid_dim, valid_dim)
            valid_flowY = central_crop(examples[k]['flowYY'], valid_dim, valid_dim)
            gt_flow_X_025 = cv2.resize(valid_flowX / 4, None, fx=0.25, fy=0.25)
            gt_flow_X_0125 = cv2.resize(valid_flowX / 8, None, fx=0.125, fy=0.125)
            gt_flow_Y_025 = cv2.resize(valid_flowY / 4, None, fx=0.25, fy=0.25)
            gt_flow_Y_0125 = cv2.resize(valid_flowY / 8, None, fx=0.125, fy=0.125)
            target_025 = np.concatenate((gt_flow_X_025[np.newaxis, np.newaxis, :, :],
                                         gt_flow_Y_025[np.newaxis, np.newaxis, :, :]), axis=1)
            target_0125 = np.concatenate((gt_flow_X_0125[np.newaxis, np.newaxis, :, :],
                                          gt_flow_Y_0125[np.newaxis, np.newaxis, :, :]), axis=1)
            if self.do_conf_factor:
                # bad_x = np.logical_or((xx + valid_flowX) > valid_dim, (xx + valid_flowX) < -1)
                # bad_y = np.logical_or((yy + valid_flowY) > valid_dim, (yy + valid_flowY) < -1)
                # valid_mask = np.logical_not(np.logical_or(bad_x, bad_y)).astype(np.float)  # positive is valid
                bad_x025 = np.logical_or((xx025 + gt_flow_X_025) >= valid_dim/4, (xx025 + gt_flow_X_025) <= -1)
                bad_y025 = np.logical_or((yy025 + gt_flow_Y_025) >= valid_dim/4, (yy025 + gt_flow_Y_025) <= -1)
                gt_mask_025 = np.logical_not(np.logical_or(bad_x025, bad_y025)).astype(np.float)  # positive is valid
                bad_x0125 = np.logical_or((xx0125 + gt_flow_X_0125) >= valid_dim / 8, (xx0125 + gt_flow_X_0125) <= -1)
                bad_y0125 = np.logical_or((yy0125 + gt_flow_Y_0125) >= valid_dim / 8, (yy0125 + gt_flow_Y_0125) <= -1)
                gt_mask_0125 = np.logical_not(np.logical_or(bad_x0125, bad_y0125)).astype(np.float)  # positive is valid
                # bad_x = np.logical_or((xx + valid_flowX) > valid_dim, (xx + valid_flowX) < -1)
                # bad_y = np.logical_or((yy + valid_flowY) > valid_dim, (yy + valid_flowY) < -1)
                # valid_mask = np.logical_not(np.logical_or(bad_x, bad_y)).astype(np.float)  #
                # gt_mask_025 = cv2.resize(valid_mask, None, fx=0.25, fy=0.25)
                # gt_mask_0125 = cv2.resize(valid_mask, None, fx=0.125, fy=0.125)
                factor_map_025[k, :, :] = np.logical_or(gt_mask_025 == 0, gt_mask_025 == 1)
                factor_map_0125[k, :, :] = np.logical_or(gt_mask_0125 == 0, gt_mask_0125 == 1)
                target_025 = np.concatenate((target_025, gt_mask_025[np.newaxis, np.newaxis, :, :]), axis=1)
                target_0125 = np.concatenate((target_0125, gt_mask_0125[np.newaxis, np.newaxis, :, :]), axis=1)

            gt_flow_025[k, :, :, :] = target_025
            gt_flow_0125[k, :, :, :] = target_0125
        D = (D - 127) / 128
        sample = dict(patches=D, gt_flow_025=gt_flow_025, gt_flow_0125=gt_flow_0125,
                      factor_map_025=factor_map_025, factor_map_0125=factor_map_0125,
                      path=input_file)
        sample = self.transform(sample)
        self.last_sample = sample
        return sample



class MatlabBasedCosegDataset(dataset.Dataset):
    def __init__(self, root_dir, example_format, do_conf_factor=True, transform_list = [ToTensor()]):
        self.example_format = example_format
        self.root_dir = Path(root_dir)
        self.example_paths = self.root_dir.glob(example_format)
        self.example_paths = [path.as_posix() for path in self.example_paths]

        self.do_conf_factor = do_conf_factor
        self.transform = transforms.Compose(transform_list)
        self.last_sample = None

    def __len__(self):
        return len(self.example_paths)

    def __getitem__(self, idx):
        input_file = self.example_paths[idx]
        # input_file = '/home/erez/Downloads/rectifiedPatchesSIFT_128X128_withField_49.mat'
        # if self.last_sample is not None:
        #     return self.last_sample
        dicts = parse_mat(spio.loadmat(input_file))
        dicts = [d for d in dicts if d['pSrc'].size > 0]
        batch_size = min(256, len(dicts))
        valid_dim = 64
        D = np.zeros((batch_size, 6, 134, 134))
        if self.do_conf_factor:
            gt_flow_025 = np.zeros((batch_size, 3, int(valid_dim / 4), int(valid_dim / 4)))
            gt_flow_0125 = np.zeros((batch_size, 3, int(valid_dim / 8), int(valid_dim / 8)))
        else:
            gt_flow_025 = np.zeros((batch_size, 2, int(valid_dim / 4), int(valid_dim / 4)))
            gt_flow_0125 = np.zeros((batch_size, 2, int(valid_dim / 8), int(valid_dim / 8)))

        factor_map_025 = np.zeros((batch_size, int(valid_dim / 4), int(valid_dim / 4)))
        factor_map_0125 = np.zeros((batch_size, int(valid_dim / 8), int(valid_dim / 8)))
        xx, yy = np.meshgrid(range(0, valid_dim), range(0, valid_dim))
        # TODO: combing 'lossFactor' with valid mask
        for k in range(0, batch_size):
            D[k, :, :, :] = np.expand_dims(np.vstack((cv2.resize(dicts[k]['pSrc'], (134, 134)).transpose((2, 0, 1)),
                                                      cv2.resize(dicts[k]['pDst'], (134, 134)).transpose((2, 0, 1)))),
                                           axis=0)
            valid_flowX = central_crop(dicts[k]['flowXX'], valid_dim, valid_dim)
            valid_flowY = central_crop(dicts[k]['flowYY'], valid_dim, valid_dim)
            gt_flow_X_025 = cv2.resize(valid_flowX / 4, None, fx=0.25, fy=0.25)
            gt_flow_X_0125 = cv2.resize(valid_flowX / 8, None, fx=0.125, fy=0.125)
            gt_flow_Y_025 = cv2.resize(valid_flowY / 4, None, fx=0.25, fy=0.25)
            gt_flow_Y_0125 = cv2.resize(valid_flowY / 8, None, fx=0.125, fy=0.125)
            target_025 = np.concatenate((gt_flow_X_025[np.newaxis, np.newaxis, :, :],
                                         gt_flow_Y_025[np.newaxis, np.newaxis, :, :]), axis=1)
            target_0125 = np.concatenate((gt_flow_X_0125[np.newaxis, np.newaxis, :, :],
                                          gt_flow_Y_0125[np.newaxis,np.newaxis, :, :]), axis=1)
            if self.do_conf_factor:
                bad_x = np.logical_or((xx + valid_flowX) > valid_dim, (xx + valid_flowX) < -1)
                bad_y = np.logical_or((yy + valid_flowY) > valid_dim, (yy + valid_flowY) < -1)
                valid_mask = np.logical_not(np.logical_or(bad_x, bad_y)).astype(np.float)  # positive is valid
                gt_mask_025 = cv2.resize(valid_mask, None, fx=0.25, fy=0.25)
                gt_mask_0125 = cv2.resize(valid_mask, None, fx=0.125, fy=0.125)
                factor_map_025[k,:,:] = np.logical_or(gt_mask_025 == 0, gt_mask_025 == 1)
                factor_map_0125[k,:,:] = np.logical_or(gt_mask_0125 == 0, gt_mask_0125 == 1)
                target_025 = np.concatenate((target_025, gt_mask_025[np.newaxis, np.newaxis, :, :]), axis=1)
                target_0125 = np.concatenate((target_0125, gt_mask_0125[np.newaxis, np.newaxis, :, :]), axis=1)

            gt_flow_025[k, :, :, :] = target_025
            gt_flow_0125[k, :, :, :] = target_0125
        D = (D - 127) / 128
        sample = dict(patches=D, gt_flow_025=gt_flow_025,gt_flow_0125=gt_flow_0125,
                      factor_map_025=factor_map_025, factor_map_0125=factor_map_0125)
        sample = self.transform(sample)
        self.last_sample = sample
        return sample