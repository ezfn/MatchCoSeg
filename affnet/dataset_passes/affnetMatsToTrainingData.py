import glob
import os
import scipy.io as sio
import cv2
from pathlib import Path
import numpy as np
import pickle as pkl

def convert_sintel(root_dir = '/media/erez/PassportRD1/SintelCleanPasses/framesAndOldPasses',speed = 1):
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
            inFilePath = I1f.replace('framesAndOldPasses', affnet_dir_name).replace('.png', '.mat')
            if not os.path.isfile(inFilePath):
                print('no such file as {}'.format(inFilePath))
                continue
            outFilePath = inFilePath.replace('.mat', '_trainData.pkl')
            src_img = cv2.imread(I1f)
            tgt_img = cv2.imread(Iotherf)
            if os.path.isfile(outFilePath):
                continue
            dict = sio.loadmat(inFilePath)
            for k in range(dict['dists'].size):
                Ha = np.matmul(np.vstack((dict['LAFS1'][k,:,:], [[0, 0, 1]])), np.linalg.pinv(np.vstack((dict['LAFS2'][k,:,:], [[0, 0, 1]]))))
            


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







if __name__ == '__main__':
    convert_sintel()






