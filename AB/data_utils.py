from torch.utils.data import dataset
from torchvision import transforms
import scipy.io as spio
import os
from pathlib import Path
import numpy as np
import cv2
import torch
# torch.multiprocessing.set_start_method("spawn")
import random
import math
from threading import Thread
from lifetobot_sdk.Geometry.coordinate_transformations import homogenous_transform_grid
from lifetobot_sdk.Geometry.image_transformations import homogenous_transform_get_patch_from_tgt
from lifetobot_sdk.Visualization import drawers as d
import csv


class ToTensor(object):
    """Convert patches in sample to Tensors."""
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, sample):

        sample['input_image'] = torch.Tensor(sample['input_image']).to(self.device)
        sample['output_map'] = torch.Tensor(sample['output_map']).to(self.device)
        sample['ellipse_params'] = torch.Tensor(sample['ellipse_params']).to(self.device)
        sample['is_ellipse'] = torch.Tensor([sample['is_ellipse']]).to(self.device)
        return sample

class ColorJitterExample(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)

    def __call__(self, example):
        """
        Args:
            example (example dict).

        Returns:
            transformed example: Color jittered image (mask untouched).
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        example['input_image'] = transform(transforms.ToPILImage(example['input_image']))
        return example


class FileBasedEllipseImageDataset(dataset.Dataset):
    def __init__(self, root_dir, train_labels_file, label_string_replacer,
                 transform_list = [ToTensor()], do_normalize_image=True):
        self.root_dir = Path(root_dir)
        with open(train_labels_file, 'r') as f:
            reader = csv.reader(f)
            train_labels = list(reader)
        self.train_labels = train_labels[1:]
        # self.example_paths = self.root_dir.glob(input_format)
        # self.example_paths = [path.as_posix() for path in self.example_paths]
        self.label_string_replacer = label_string_replacer
        self.do_visualize = False
        self.transform = transforms.Compose(transform_list)
        self.do_normalize_image = do_normalize_image

    def __len__(self):
        return len(self.train_labels)
        # return len(self.example_paths)

    def reorder_ellipse_params(self, ellipse_params):
        '''
        :param ellipse_params: (cx,cy,theta,half_ax1,half_ax2)
        :return: reordered_ellipse_params: (cx,cy,theta,half_major_ax,half_minor_ax)
        '''
        if ellipse_params[4] > ellipse_params[3]:
            reordered_ellipse_params = np.hstack((ellipse_params[0:2],ellipse_params[2], ellipse_params[4],
                                                  ellipse_params[3]))
        else:
            reordered_ellipse_params = np.hstack((ellipse_params[0:2], ellipse_params[2], ellipse_params[3:5]))
        return reordered_ellipse_params

    def __getitem__(self, idx):
        label = self.train_labels[idx]
        ellipse_params = np.array(list(map(float, label[1:])))
        is_ellipse = os.path.join(label[0].split(' ')[1]) == 'True'
        if is_ellipse:
            ellipse_params = self.reorder_ellipse_params(ellipse_params)
        else:
            ellipse_params = np.ones_like(ellipse_params) # avoid zeros
        input_file = os.path.join(self.root_dir.as_posix(), label[0].split(' ')[0])
        Iin = cv2.imread(input_file).transpose((2, 0, 1))
        out_map_file = self.label_string_replacer(input_file)
        out_map = cv2.imread(out_map_file).astype(np.float).transpose((2, 0, 1))[None, 0, :, :]
        # input_file = self.example_paths[idx]
        # Iin = cv2.imread(input_file).astype(np.float).transpose((2, 0, 1))
        # out_map_file = self.label_string_replacer(input_file)
        # out_map = cv2.imread(out_map_file).astype(np.float).transpose((2, 0, 1))[None, 0, :,:]
        if self.do_normalize_image:
            Iin = (Iin - 127) / 128
        out_map = out_map/np.maximum(np.max(out_map), 1)
        example = dict(input_image=Iin, output_map=out_map, ellipse_params=ellipse_params, is_ellipse=is_ellipse)
        example = self.transform(example)
        return example