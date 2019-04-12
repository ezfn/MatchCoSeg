import glob
import os
from affnet.api import affine_adapted_matcher
from lifetobot_sdk.Geometry import image_transformations
from lifetobot_sdk.Visualization import drawers as d
import scipy.io as sio
import cv2
dirs = glob.glob('/media/rd/MyPassport/CoSegDataPasses/MPI_SINTEL_7Z/training/clean/*')
aff_matcher = affine_adapted_matcher.AffineMatcher(do_use_cuda=False)
speed = 1
output_dir_name = 'affnet_{}'.format(speed)
bad_list = []
for currentDir in dirs:
    print('In directory: ' + currentDir)
    imageFiles = sorted(glob.glob(currentDir + '/frame_*.png'))
    output_dir = currentDir.replace('clean',output_dir_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for imgIdx in range(0, imageFiles.__len__()-speed):
        I1f = imageFiles[imgIdx]
        Iotherf = imageFiles[imgIdx+speed]
        outFilePath = I1f.replace('clean',output_dir_name).replace('.png', '.mat')
        src_img = cv2.imread(I1f)
        tgt_img = cv2.imread(Iotherf)
        if os.path.isfile(outFilePath):
            continue
        try:
            out_dict, P1, P2 = aff_matcher.match_images(src_img, tgt_img)
        except:
            bad_list.append(imgIdx)
            print('failed to run matcher for the {} time'.format(len(bad_list)))
            continue
        save_dict = dict(dists=out_dict['dists'], LAFS1=out_dict['LAFs1'].cpu().numpy(),
                         LAFS2=out_dict['LAFs2'].cpu().numpy())
        sio.savemat(outFilePath, save_dict)
        print('matches saved to ' + outFilePath)
