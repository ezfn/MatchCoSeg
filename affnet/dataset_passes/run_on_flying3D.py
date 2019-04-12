from affnet.api import affine_adapted_matcher
from lifetobot_sdk.Geometry import image_transformations
from lifetobot_sdk.Visualization import drawers as d
import cv2
import scipy.io as sio
import glob
import os
import numpy as np
speed = 3
aff_matcher = affine_adapted_matcher.AffineMatcher(do_use_cuda=True)
root_dir = '/media/rd/MyPassport/CoSegDataPasses/flyingThings3D_FLOWNET_split/FlyingThings3D_subset/val'
flow_dir = os.path.join(root_dir, 'flow/left/into_future_{}'.format(speed))
occ_dir = os.path.join(root_dir, 'flow_occlusions/left/into_future_{}'.format(speed))
png_dir = os.path.join(root_dir, 'image_clean/left')
output_dir = os.path.join(root_dir, 'affnet/left_{}'.format(speed))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

png_list = glob.glob(os.path.join(png_dir,'*.png'))
flow_list = glob.glob(os.path.join(flow_dir,'*.flo'))
occ_list = glob.glob(os.path.join(occ_dir,'*.png'))

nums_flow = [f.split('/')[-1].split('.')[0] for f in flow_list]
last_idx = np.max([int(s) for s in nums_flow])
last_idx = 10*((last_idx+1)//10) - 1
all_nums = np.arange(0,last_idx+1)
list_of_groups = list(zip(*(iter(all_nums),) * (10)))


nums_occ = [f.split('/')[-1].split('.')[0] for f in occ_list]
nums_png = [f.split('/')[-1].split('.')[0] for f in png_list]
# meta_set = set(nums_occ).intersection(set(nums_flow))
# valid_nums = list(set(nums_png).intersection(meta_set))
# valit_ints = [int(s) for s in valid_nums]
# valit_ints = sorted(valit_ints)
# list_of_groups = list(zip(*(iter(valit_ints),) * (9-speed+1)))
bad_list = []
# for group in list_of_groups:
for img_num in all_nums:
    if '%07d' % img_num not in nums_png or '%07d' % img_num not in nums_flow:
        continue
    outFilePath = os.path.join(output_dir, 'affnetMatches_' + '%07d' % img_num + '.mat')
    if os.path.isfile(outFilePath):
        continue
    src_img = cv2.imread(os.path.join(png_dir, '%07d' % img_num + '.png'))
    tgt_img = cv2.imread(os.path.join(png_dir, '%07d' % (img_num+speed) + '.png'))
    try:
        out_dict, P1, P2 = aff_matcher.match_images(src_img, tgt_img)
    except:
        bad_list.append(img_num)
        print('failed to run matcher for the {} time'.format(len(bad_list)))
        continue
    save_dict = dict(dists=out_dict['dists'], LAFS1=out_dict['LAFs1'].cpu().numpy(),
                     LAFS2=out_dict['LAFs2'].cpu().numpy())
    sio.savemat(outFilePath, save_dict)
    print('matches saved to ' + outFilePath)
print(bad_list)



# I1 = cv2.imread('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/img1.png')
# I2 = cv2.imread('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/img6.png')
# out_file = ('/media/Media/SWDEV/repos/MatchCoSeg/affnet/test-graf/matches.mat')
# d.imshow(I2 ,wait_time=1)
# out_dict ,P1 ,P2 = aff_matcher.match_images(I1 ,I2)
# save_dict = dict(dists=out_dict['dists'], LAFS1=out_dict['LAFs1'].cpu().numpy(), LAFS2=out_dict['LAFs2'].cpu().numpy())
# sio.savemat(out_file, save_dict)