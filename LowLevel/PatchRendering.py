import numpy as np
import cv2
from lifetobot_sdk.Geometry import coordinate_transformations as geom_trans

def get_affine_templates(I, src_pts, template_extent, Ha):

    XX0,YY0 = np.meshgrid(range(-template_extent,template_extent+1), range(-template_extent,template_extent+1))
    XXT = np.zeros((XX0.shape[0],XX0.shape[1]*src_pts.shape[0]))
    YYT = np.zeros((YY0.shape[0],YY0.shape[1]*src_pts.shape[0]))
    dst_points = geom_trans.do_affine_transform(src_pts, Ha)
    for k in range(src_pts.shape[0]):
        XXT[:,(XX0.shape[1]*k):(XX0.shape[1]*(k+1))] = XX0+dst_points[k,0]
        YYT[:, (YY0.shape[1] * k):(YY0.shape[1] * (k + 1))] = YY0 + dst_points[k, 1]
    [Xq, Yq] = geom_trans.affine_transform_grid(XXT, YYT, geom_trans.get_Affine_inv(Ha))
    #TODO: cuda/pytorch this
    interpedData = cv2.remap(I, Xq.astype(np.float32), Yq.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    templates = np.hsplit(interpedData, src_pts.shape[0])
    bad_ind = [np.any(np.isnan(T)) for T in templates]
    return templates, dst_points, bad_ind