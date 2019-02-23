import torch.nn as nn

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)

def predict_flow_and_conf(in_planes):
    return nn.Conv2d(in_planes, 3, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


def context_network(in_planes):
    out_planes = 32
    return nn.Sequential(
        conv(in_planes, 128, kernel_size=3, stride=1, padding=1, dilation=1),
        conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
        conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
        conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8),
        conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16),
        conv(64, out_planes, kernel_size=3, stride=1, padding=1, dilation=1)
    ), out_planes