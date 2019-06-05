from matplotlib import pyplot as plt
from lifetobot_sdk.Visualization import drawers
from affnet.dataset_passes import flowlib
import numpy as np
def plot_3ch_kernels(tensor, num_cols=6):
    tensor = tensor.detach().cpu().numpy()
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    return fig

def show_3ch_pairs(tensor1, tensor2, num_cols=6):
    tensor1 = tensor1.detach().cpu().numpy()
    tensor2 = tensor2.detach().cpu().numpy()
    if not (tensor1.ndim==4 and tensor2.ndim==4):
        raise Exception("assumes a 4D tensor")
    if not (tensor1.shape[-1]==3 and tensor2.shape[-1]==3):
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor1.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor1.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor1[i]*0.5 + tensor2[i]*0.5)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    return fig

def imshow_tensor(tensor, num_cols=6):
    tensor = tensor.detach().cpu().numpy()
    if not tensor.ndim == 3:
        raise Exception("assumes a 3D tensor")
    # if not tensor.shape[-1] == 3:
    #     raise Exception("last dim needs to be 3 to plot")
    num_channels = tensor.shape[0]
    num_rows = 1 + num_channels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(num_channels):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    return fig

def imshow_dual_image(tensor):
    tensor = tensor.detach().cpu().numpy()
    if not tensor.ndim == 3:
        raise Exception("assumes a 3D tensor")
    fig = plt.figure(figsize=(2, 1))
    for i in range(2):
        ax1 = fig.add_subplot(1, 2, i + 1)
        ax1.imshow(tensor[(i*3):((i+1)*3), :, :].transpose(1,2,0))
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    return fig

def multi_imshow_tensor(tensor):
    tensor = tensor.detach().cpu().numpy()
    if not tensor.ndim == 4:
        raise Exception("assumes a 4D tensor")
    # if not tensor.shape[-1] == 3:
    #     raise Exception("last dim needs to be 3 to plot")
    num_channels = tensor.shape[1]
    num_cols = num_channels
    num_rows = tensor.shape[0]
    fig = plt.figure(figsize=(num_channels, num_rows))
    for r in range(num_rows):
        for c in range(num_cols):
            ax1 = fig.add_subplot(num_rows, num_cols, r*num_cols + c + 1)
            ax1.imshow(tensor[r,c,:,:])
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
    return fig

def multi_flow_show_tensor(tensor, flow_losses=None):
    tensor = tensor.detach().cpu().numpy()
    if not tensor.ndim == 4:
        raise Exception("assumes a 4D tensor")
    # if not tensor.shape[-1] == 3:
    #     raise Exception("last dim needs to be 3 to plot")
    num_cols = 2
    num_rows = tensor.shape[0]
    fig = plt.figure(figsize=(num_cols, num_rows))
    for r in range(num_rows):
        ax1 = fig.add_subplot(num_rows, num_cols, r * num_cols + 1)
        flow_img = flowlib.flow_to_image(tensor[r, 0:2, :, :].transpose(1,2,0))
        ax1.imshow(flow_img)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        if flow_losses is not None:
            ax1.set_title("flow_loss: " + str(flow_losses[r]), fontsize=10)
        ax1 = fig.add_subplot(num_rows, num_cols, r * num_cols + 2)
        ax1.imshow(tensor[r, 2, :, :])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    return fig