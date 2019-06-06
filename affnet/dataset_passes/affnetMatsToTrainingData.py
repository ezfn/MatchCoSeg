import glob
import os
import scipy.io as sio
import cv2
from pathlib import Path
import numpy as np
from affnet.dataset_passes.flyingThingsUtils import readFlow
import pickle as pkl
from flowNet.data_utils import produce_warped_example
import gzip


def convert_sintel(root_dir = '/media/rd/MyPassport/CoSegDataPasses/SintelCleanPasses/framesAndOldPasses',speed = 1,
                   patch_shape=[134,134], max_examples_per_image = 500):
    from lifetobot_sdk.Geometry import image_transformations
    from lifetobot_sdk.Visualization import drawers as d
    root_path = Path(root_dir)
    dirs = root_path.glob('*')
    affnet_dir_name = 'affnet_{}'.format(speed)
    flow_dir_name = 'flow_{}'.format(speed)
    occ_dir_name = 'occlusions_{}'.format(speed)

    for currentDir in dirs:
        currentDir = currentDir.as_posix()
        print('In directory: ' + currentDir)
        imageFiles = sorted(glob.glob(currentDir + '/frame_*.png'))
        affnet_dir = currentDir.replace('framesAndOldPasses', affnet_dir_name)
        flow_dir = currentDir.replace('framesAndOldPasses', flow_dir_name)
        occ_dir = currentDir.replace('framesAndOldPasses', occ_dir_name)
        if not os.path.isdir(affnet_dir):
            print('no such dir as {}'.format(affnet_dir))
        for imgIdx in range(0, imageFiles.__len__() - speed):
            I1f = imageFiles[imgIdx]
            Iotherf = imageFiles[imgIdx + speed]
            flow_file = I1f.replace('framesAndOldPasses', flow_dir_name).replace('.png', '.flo')
            occ_file = I1f.replace('framesAndOldPasses', occ_dir_name)
            inFilePath = I1f.replace('framesAndOldPasses', affnet_dir_name).replace('.png', '.mat')
            if not os.path.isfile(inFilePath):
                print('no such file as {}'.format(inFilePath))
                continue
            outFilePath = inFilePath.replace('.mat', '_trainData.pklz')
            src_img = cv2.imread(I1f)
            tgt_img = cv2.imread(Iotherf)
            flow = readFlow(flow_file)
            occ_mask = cv2.imread(occ_file)
            xx,yy = np.meshgrid(range(0,src_img.shape[1]), range(0,src_img.shape[0]))
            xx_tgt = xx + flow[:,:,0]
            yy_tgt = yy + flow[:, :, 1]
            xx_tgt[occ_mask[:, :, 0] > 0] = -10000
            yy_tgt[occ_mask[:, :, 0] > 0] = -10000
            flow_dict = dict(xx_tgt=xx_tgt, yy_tgt=yy_tgt, src_img=src_img, tgt_img=tgt_img, occ_mask=occ_mask)
            if os.path.isfile(outFilePath) and False:
                continue
            affnet_dict = sio.loadmat(inFilePath)
            examples = []
            chosen_idxs = np.argsort(affnet_dict['dists'].flatten())[0:min(max_examples_per_image, affnet_dict['dists'].size)]
            for k in chosen_idxs:
                Ha = np.matmul(np.vstack((affnet_dict['LAFS1'][k,:,:], [[0, 0, 1]])), np.linalg.pinv(np.vstack((affnet_dict['LAFS2'][k,:,:], [[0, 0, 1]]))))
                srcCenter = np.squeeze(affnet_dict['LAFS1'][k, 0:2, 2])
                example = produce_warped_example(srcCenter, Ha, patch_shape, flow_dict)
                if example is not None:
                    examples.append(example)
            with gzip.open(outFilePath,'wb') as f:
                pkl.dump(examples, f)
            print('saved {} to disk with {} examples'.format(outFilePath, len(examples)))


if __name__ == '__main__':
    convert_sintel(speed=1)






