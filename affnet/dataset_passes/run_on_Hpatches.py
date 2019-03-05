from affnet.api import affine_adapted_matcher
from lifetobot_sdk.Geometry import image_transformations
from lifetobot_sdk.Visualization import drawers as d
import cv2
import scipy.io as sio
aff_matcher = affine_adapted_matcher.AffineMatcher(do_use_cuda=True)

I1 = cv2.imread('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/img1.png')
I2 = cv2.imread('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/img6.png')
out_file = ('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/matches.mat')
d.imshow(I2 ,wait_time=1)
out_dict ,P1 ,P2 = aff_matcher.match_images(I1 ,I2)
save_dict = dict(dists=out_dict['dists'], LAFS1=out_dict['LAFs1'].cpu().numpy(), LAFS2=out_dict['LAFs2'].cpu().numpy())
sio.savemat(out_file, save_dict)