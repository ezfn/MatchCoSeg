from AB.mobilenet.models.imagenet import mobilenetv2,ellipse_detection_model
import torch
import cv2
import numpy as np
# import importlib
# importlib.reload(mobilenetv2)
from Visualization.torch_vis import imshow_tensor
from matplotlib import pyplot as plt

net = ellipse_detection_model.Detector(mobilenet_pre_trained_weights='/Umedia/SWDEV/CoSegMatching/AB/mobilenet/pretrained/mobilenetv2_1.0-0c6065bc.pth')
net.to('cuda')
I = cv2.imread('/home/rd/Downloads/images/test/0019.jpg')
I = I.transpose((2, 0, 1))
T = torch.Tensor(np.expand_dims(I, axis=0)).to('cuda')
mask_logits, class_logit = net(T)

# def run_first_n_layers(I, n_layers):
#     for k in range(n_layers):
#         I = net.features._modules[str(k)](I)
#     return I




# net = mobilenetv2()
# net.load_state_dict(torch.load('/Umedia/SWDEV/CoSegMatching/AB/mobilenet/pretrained/mobilenetv2_1.0-0c6065bc.pth'))
#
# I = cv2.imread('/home/rd/Downloads/images/test/0019.jpg')
# I = I.transpose((2, 0, 1))
# out = run_first_n_layers(torch.Tensor(np.expand_dims(I, axis=0)), 5)
#
# imshow_tensor(out[0,:,:,:])
#
# out_np = out.to
#
#
# feature_map = net.get_feature_map(torch.Tensor(np.expand_dims(I, axis=0)))
#
#
#
