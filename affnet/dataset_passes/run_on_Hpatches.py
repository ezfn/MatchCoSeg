from affnet.api import affine_adapted_matcher
from lifetobot_sdk.Geometry import image_transformations
from lifetobot_sdk.Visualization import drawers as d
import cv2
import scipy.io as sio
import glob
import os
aff_matcher = affine_adapted_matcher.AffineMatcher(do_use_cuda=False)
dirs = glob.glob('/media/rd/MyPassport/CoSegDataPasses/hpatches-sequences-release/v_*')
bad_dirs = [21]
for currentDir in dirs[22:]:
    print('In directory: ' + currentDir)
    I1f = os.path.join(currentDir, '1.ppm')
    for otherImgIdx in range(2, 7):
        Iotherf = os.path.join(currentDir, str(otherImgIdx) + '.ppm')
        outFilePath = os.path.join(currentDir, 'affnetMatches_' + str(otherImgIdx) + '.mat')
        out_dict, P1, P2 = aff_matcher.match_images(cv2.imread(I1f), cv2.imread(Iotherf))
        save_dict = dict(dists=out_dict['dists'], LAFS1=out_dict['LAFs1'].cpu().numpy(),
                         LAFS2=out_dict['LAFs2'].cpu().numpy())
        sio.savemat(outFilePath, save_dict)
        print('matches saved to ' + outFilePath)





# I1 = cv2.imread('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/img1.png')
# I2 = cv2.imread('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/img6.png')
# out_file = ('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/matches.mat')
# d.imshow(I2 ,wait_time=1)
# out_dict ,P1 ,P2 = aff_matcher.match_images(I1 ,I2)
# save_dict = dict(dists=out_dict['dists'], LAFS1=out_dict['LAFs1'].cpu().numpy(), LAFS2=out_dict['LAFs2'].cpu().numpy())
# sio.savemat(out_file, save_dict)