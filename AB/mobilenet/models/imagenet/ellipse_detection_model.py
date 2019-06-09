import torch
import torch.nn as nn
from AB.mobilenet.models.imagenet import mobilenetv2
import math


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def excitation(inp, oup, ratio):
    return nn.Sequential(
        nn.Linear(inp, inp // ratio, bias=True),
        nn.ReLU6(inplace=True),
        nn.Linear(inp // ratio, oup, bias=True),
        nn.Sigmoid()
    )

def upsize_conv(in_planes, out_planes, out_size, kernel_size=5, padding=2):
    return nn.Sequential(
        nn.UpsamplingNearest2d(size=out_size),
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding)
    )

def params_estimator(in_dims, out_dims):
    return nn.Sequential(
        nn.Linear(in_dims, in_dims // 2),
        nn.LeakyReLU(inplace=True),
        nn.Linear(in_dims // 2, 5)
    )




class Detector(nn.Module):
    def __init__(self, mobilenet_pre_trained_weights=None, output_mask_size = 50, n_mobilenet_layers_to_run = 5):
        super(Detector, self).__init__()
        self.mobile_net = mobilenetv2()
        if mobilenet_pre_trained_weights is not None:
            self.mobile_net.load_state_dict(
                torch.load(mobilenet_pre_trained_weights))
        self.n_mobilenet_layers_to_run = n_mobilenet_layers_to_run
        self.feature_map_channels = 32
        self.feature_map_size = 7
        self.avgpool = nn.AvgPool2d(self.feature_map_size, stride=1)
        self.channel_conv = conv_1x1_bn(self.feature_map_channels, self.feature_map_channels)
        self.produce_mask_logits = nn.Conv2d(self.feature_map_channels // 4, 1, 1, 1, 0, bias=False)
        self.excitation_layer = excitation(self.feature_map_channels, self.feature_map_channels, 4)
        self.classifier = nn.Linear(self.feature_map_channels, 1)
        self.params_estimator = params_estimator(self.feature_map_channels, 5)
        self.upsizer1 = upsize_conv(self.feature_map_channels, self.feature_map_channels//2, (output_mask_size//2, output_mask_size//2))
        self.upsizer2 = upsize_conv(self.feature_map_channels//2, self.feature_map_channels // 4,
                                    (output_mask_size, output_mask_size))

    def estimate_params(self, input_vec):
        params = self.params_estimator(input_vec)
        params = torch.cat((params[:, :, :, 0:2], (180*torch.atan(params[:, :, :, 2:3]))/math.pi, params[:, :, :, 3:5]), dim=3)
        return params


    def forward(self, x):
        # standard convolutions
        x = self.mobile_net.run_first_n_layers(x, self.n_mobilenet_layers_to_run)
        x = self.channel_conv(x)
        # excitation from global information
        x_pooled = self.avgpool(x)
        x_pooled = x_pooled.permute(0, 2, 3, 1)
        excite_vec = self.excitation_layer(x_pooled)
        excite_vec = excite_vec.permute(0, 3, 1, 2)
        x = x * excite_vec
        # classification
        class_logit = self.classifier(x_pooled)
        param_logits = self.estimate_params(x_pooled).transpose(1, 3)
        # upsizing for mask
        x = self.upsizer1(x)
        x = self.upsizer2(x)
        # mask production
        mask_logits = self.produce_mask_logits(x)
        return mask_logits, class_logit, param_logits

