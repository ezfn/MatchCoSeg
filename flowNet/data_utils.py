
from torch.utils.data import dataset
from torchvision import transforms
import scipy.io as spio
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import tensorflow as tf
from flowNet import recordReader
import random
import math
from threading import Thread
from lifetobot_sdk.Geometry.coordinate_transformations import homogenous_transform_grid
from lifetobot_sdk.Geometry.image_transformations import homogenous_transform_get_patch_from_tgt
from lifetobot_sdk.Visualization import drawers as d





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



class tfRecordBasedCosegDataset(dataset.Dataset):


    def __init__(self, record_path, buffer_size=1000, transform_list = [ToTensor()]):
        self.transform = transforms.Compose(transform_list)
        example_names = dict(src_patch='patches/src', tgt_patch='patches/tgt',
                             gt_flow_0125='flow/gt_flow_0125',gt_flow_025='flow/gt_flow_025',
                             factor_map_0125='flow/factor_map_0125', factor_map_025='flow/factor_map_025',
                             path='meta/path',sampled_idxs='meta/sampled_idx')
        self.record_reader = recordReader.recordReader(filename=record_path, example_dict=example_names)
        self.buffer = []
        self.exposed_buffer_size = buffer_size
        self.full_buffer_size = buffer_size*5
        self._fill_up_buffer()
        self._is_filling_lock = False

    def __len__(self):
        return self.exposed_buffer_size


    def _fill_up_buffer(self):
        self._is_filling_lock = True
        n_to_fill = self.full_buffer_size - len(self.buffer)
        for k in range(n_to_fill):
            self.buffer.append(self._get_next())
        random.shuffle(self.buffer)
        self._is_filling_lock = False

    def _get_next(self):
        parsed_example = self.record_reader.get_next_example_parsed()
        while np.any(np.isnan(parsed_example['gt_flow_0125'])) or np.any(np.isnan(parsed_example['gt_flow_025'])):
            print('skipping bad example in path {}'.format(parsed_example['path']))
            parsed_example = self.record_reader.get_next_example_parsed()
        patches = np.vstack((parsed_example['src_patch'].transpose((2, 0, 1)),
                                            parsed_example['tgt_patch'].transpose((2, 0, 1))))
        sample = dict(patches=patches, gt_flow_025=parsed_example['gt_flow_025'],
                      gt_flow_0125=parsed_example['gt_flow_0125'],
                      factor_map_025=parsed_example['factor_map_025'],
                      factor_map_0125=parsed_example['factor_map_0125'],
                      path=parsed_example['path'], sampled_idxs=parsed_example['sampled_idxs'])
        return self.transform(sample)

    def get_next(self):
        parsed_example = self.record_reader.get_next_example_parsed()
        patches = np.expand_dims(np.vstack((parsed_example['src_patch'].transpose((2, 0, 1)),
                                            parsed_example['tgt_patch'].transpose((2, 0, 1)))),
                                 axis=0)
        sample = dict(patches=patches, gt_flow_025=np.expand_dims(parsed_example['gt_flow_025'], 0),
                      gt_flow_0125=np.expand_dims(parsed_example['gt_flow_0125'], 0),
                      factor_map_025=np.expand_dims(parsed_example['factor_map_025'], 0),
                      factor_map_0125=np.expand_dims(parsed_example['factor_map_0125'], 0),
                      path=parsed_example['path'], sampled_idxs=parsed_example['sampled_idxs'])
        return self.transform(sample)

    def get_batch(self, batch_size=16):
        batch = dict(patches=[], gt_flow_025=[],
                      gt_flow_0125=[],
                      factor_map_025=[],
                      factor_map_0125=[],
                      path=[], sampled_idxs=[])
        for k in range(batch_size):
            sample = self.__getitem__(k)
            for key in sample.keys():
                batch[key].append(sample[key])
        for key in batch.keys():
            if key not in ['path', 'sampled_idxs']:
                batch[key] = torch.stack(batch[key],dim=0)
        return batch


    def __getitem__(self, idx):
        #maybe wait for lock
        while (self._is_filling_lock):
            continue
        sample = self.buffer[idx]
        self.last_sample = sample
        del self.buffer[idx]
        if len(self.buffer) <= self.exposed_buffer_size:
            thread = Thread(target=self._fill_up_buffer())
            thread.start()
        return sample

        # sample = dict(patches=D, gt_flow_025=gt_flow_025, gt_flow_0125=gt_flow_0125,
        #               factor_map_025=factor_map_025, factor_map_0125=factor_map_0125)
        pass


class PickleBasedCosegDataset(dataset.Dataset):
    def __init__(self, root_dir, example_format, do_conf_factor=True, transform_list = [ToTensor()],
                 max_batch_size = 256, do_normalize_image=True):
        self.example_format = example_format
        self.root_dir = Path(root_dir)
        self.example_paths = self.root_dir.glob(example_format)
        self.example_paths = [path.as_posix() for path in self.example_paths]
        self.max_batch_size = max_batch_size
        self.do_visualize = False

        self.do_conf_factor = do_conf_factor
        self.transform = transforms.Compose(transform_list)
        self.last_sample = None
        self.do_normalize_image = do_normalize_image

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
        batch_size = min(self.max_batch_size, len(examples))
        valid_dim = 64
        fringe = (134 - valid_dim)/2
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
        sampled_idxs = np.random.choice(len(examples), size=batch_size, replace=False)
        for k, idx in enumerate(sampled_idxs):
            example = examples[idx]
            D[k, :, :, :] = np.expand_dims(np.vstack((example['src_patch'].transpose((2, 0, 1)),
                                                      example['tgt_patch'].transpose((2, 0, 1)))),
                                           axis=0)
            valid_flowX = central_crop(example['flowXX'], valid_dim, valid_dim)
            valid_flowY = central_crop(example['flowYY'], valid_dim, valid_dim)
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
                bad_x025 = np.logical_or((xx025 + gt_flow_X_025) >= (valid_dim+fringe)/4, (xx025 + gt_flow_X_025) <= -fringe/4)
                bad_y025 = np.logical_or((yy025 + gt_flow_Y_025) >= (valid_dim+fringe)/4, (yy025 + gt_flow_Y_025) <= -fringe/4)
                gt_mask_025 = np.logical_not(np.logical_or(bad_x025, bad_y025)).astype(np.float)  # positive is valid
                bad_x0125 = np.logical_or((xx0125 + gt_flow_X_0125) >= (valid_dim+fringe) / 8, (xx0125 + gt_flow_X_0125) <= -fringe/8)
                bad_y0125 = np.logical_or((yy0125 + gt_flow_Y_0125) >= (valid_dim+fringe) / 8, (yy0125 + gt_flow_Y_0125) <= -fringe/8)
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
            if self.do_visualize:
                from lifetobot_sdk.Visualization import drawers as d
                cropped_tgt = central_crop(example['tgt_patch'], valid_dim, valid_dim)
                src = cv2.remap(cropped_tgt, (xx + valid_flowX).astype(np.float32), (yy + valid_flowY).astype(np.float32), cv2.INTER_LINEAR)

            gt_flow_025[k, :, :, :] = target_025
            gt_flow_0125[k, :, :, :] = target_0125
        if self.do_normalize_image:
            D = (D - 127) / 128
        sample = dict(patches=D, gt_flow_025=gt_flow_025, gt_flow_0125=gt_flow_0125,
                      factor_map_025=factor_map_025, factor_map_0125=factor_map_0125,
                      path=input_file, sampled_idxs=sampled_idxs)
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

def produce_warped_example(srcCenter, Ha, patchSize, flow_dict, do_produce_gt=True):
    # targetImageSize = size(flowExample.Idst);
    xStart = int(srcCenter[0] - patchSize[1] / 2)
    xEnd = xStart + patchSize[1]
    yStart = int(srcCenter[1] - patchSize[0] / 2)
    yEnd = yStart + patchSize[0]
    if np.logical_or.reduce((yStart<0, xStart<0, yEnd > flow_dict['xx_tgt'].shape[0], xEnd > flow_dict['xx_tgt'].shape[1])):
        return None
    Xp_dst = flow_dict['xx_tgt'][yStart:yEnd, xStart:xEnd]
    Yp_dst = flow_dict['yy_tgt'][yStart:yEnd, xStart:xEnd]
    # Xq, Yq = np.meshgrid(range(xStart,xEnd), range(yStart,yEnd))
    patch_from_tgt, XXproj, YYproj = homogenous_transform_get_patch_from_tgt(flow_dict['tgt_img'],Ha, [xStart,yStart,xEnd,yEnd])
    srcPatch = flow_dict['src_img'][yStart:yEnd, xStart:xEnd, :]
    patch_from_tgt[np.isnan(patch_from_tgt)] = 127
    if do_produce_gt:
        srcMask = flow_dict['occ_mask'][yStart:yStart+patchSize[0], xStart:xStart+patchSize[1]]
        flowXX = Xp_dst - XXproj
        flowYY = Yp_dst - YYproj
        srcMask = np.logical_not(np.logical_or(srcMask[:, :, 0], np.isnan(patch_from_tgt[:, :, 0])))
        grad_metric = get_gradient_metrics(srcPatch, 24)
        loss_factor = srcMask*grad_metric
        example = dict(flowXX=flowXX, flowYY=flowYY, srcMask=srcMask, lossFactor=loss_factor,
                       src_patch=srcPatch, tgt_patch=patch_from_tgt, Ha=Ha, src_center=srcCenter)
    else:
        example = dict(flowXX=None, flowYY=None, srcMask=None, lossFactor=None,
                       src_patch=srcPatch, tgt_patch=patch_from_tgt, Ha=Ha, src_center=srcCenter)
    return example



def get_gradient_metrics(I,extent):
    xx = np.array(range(-extent,extent+1))
    gX = np.exp(-xx**2 / (0.5 * (extent **2)))
    gX = gX / np.sum(gX);
    kernel = np.outer(gX,gX)
    If = cv2.filter2D(I, I.shape[2], kernel)
    return np.sum(If,axis=2)
