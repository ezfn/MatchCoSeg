import numpy as np
import cv2
from lifetobot_sdk.Geometry import coordinate_transformations as geom_trans

def get_affine_templates(I, src_pts, template_extent, Ha):

    XX0,YY0 = np.meshgrid(range(-template_extent,template_extent+1), range(-template_extent,template_extent+1))
    tgt_points = geom_trans.do_affine_transform(src_pts, Ha)
    vfunc = np.vectorize(np.add, doc='Vectorized `myfunc`')
    vfunc(np.repeat(XX0, src_pts.shape[0], axis=1), np.repeat(tgt_points[:,0,None],XX0.shape[1],axis=1))

    tgt_X_points = np.repeat(XX0, src_pts.shape[0], axis=1)