import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from AB import fit_ellipse
import importlib
importlib.reload(fit_ellipse)
import os
import glob
import csv

root_dir = '/home/rd/Downloads/'
train_dir = os.path.join(root_dir, 'images/train')
test_dir = os.path.join(root_dir, 'images/test')
train_image_files = glob.glob(os.path.join(train_dir,'*.jpg'))
test_image_files = glob.glob(os.path.join(test_dir,'*.jpg'))
test_labels_file = os.path.join(root_dir, 'images/test_data.txt')
train_labels_file = os.path.join(root_dir, 'images/train_data.txt')
with open(train_labels_file, 'r') as f:
    reader = csv.reader(f)
    train_labels = list(reader)
train_labels = train_labels[1:]

with open(test_labels_file, 'r') as f:
    reader = csv.reader(f)
    test_labels = list(reader)
test_labels = test_labels[1:]

def fit_format_to_data_format(ellipse_tuple):
    center = list(ellipse_tuple[0])
    if ellipse_tuple[1][0] < ellipse_tuple[1][1]:
        axes = [ellipse_tuple[1][1]/2,ellipse_tuple[1][0]/2]
        angle = [(ellipse_tuple[2] + 90) % 360]
    else:
        axes = [ellipse_tuple[1][0] / 2, ellipse_tuple[1][1] / 2]
        angle = [ellipse_tuple[2]]
    return np.array(center + angle + axes)

def get_ellipse_params_err(ellipse_est, ellipse_gt):
    center_err = np.linalg.norm(ellipse_est[0:2] - ellipse_gt[0:2], ord=2)
    axis_err = np.linalg.norm(ellipse_est[3:5] - ellipse_gt[3:5], ord=2)
    angle_error_raw = ellipse_est[2] - ellipse_gt[2]
    angle_error = np.abs((angle_error_raw + 90) % 180 - 90)
    # angle_error_prcnt = angle_error / (angle_error + np.abs(ellipse_gt[2] % 180))

    return np.hstack((center_err, axis_err, angle_error))

all_errors = np.zeros((0,3))
for label in test_labels:
    ellipse_params = np.array(list(map(float, label[1:])))
    I = cv2.imread(os.path.join(root_dir, label[0].split(' ')[0]))
    is_ellipse = os.path.join(label[0].split(' ')[1])
    edges = cv2.Canny(I, 100, 200)
    cords = np.asarray(np.column_stack(np.where(edges > 0)))
    cords = np.fliplr(cords)
    # fig = plt.imshow(I)
    # plt.scatter(cords[:, 0], cords[:, 1]);
    # plt.show()
    if is_ellipse == 'True':
        best_ellipse = fit_ellipse.FitEllipse_RANSAC_Support(cords, cv2.cvtColor(I, cv2.COLOR_RGB2GRAY), None, max_itts=500,
                                                         max_refines=50, max_perc_inliers=100)
        if best_ellipse[1][0] < 1:
            best_ellipse = fit_ellipse.FitEllipse_RANSAC(cords, cv2.cvtColor(I, cv2.COLOR_RGB2GRAY), None,
                                                                 max_itts=500,
                                                                 max_refines=50, max_perc_inliers=100)
            if best_ellipse[1][0] < 1:
                best_ellipse = fit_ellipse.FitEllipse_RobustLSQ(cords, cv2.cvtColor(I, cv2.COLOR_RGB2GRAY), None,
                                              max_refines=50, max_perc_inliers=100)

        ellipse_est = fit_format_to_data_format(best_ellipse)
        errs = get_ellipse_params_err(ellipse_est, ellipse_params)
        print('center error(px.):{}, axis_error(px.):{}, angular_error(deg):{}'.format(errs[0], errs[1], errs[2]))
        if errs[0] > 20 or errs[1] > 20:
            fig = plt.imshow(I)
            plt.scatter(cords[:, 0], cords[:, 1]);
            plt.show()
        all_errors = np.vstack((all_errors, errs.reshape(1, 3)))
    plt.close('all')
pass

# I = cv2.imread('/media/Media/MEDIA/datasets/images/train/0415.jpg')
# plt.imshow(I);plt.show()
# edges = cv2.Canny(I,100,200)
# plt.imshow(edges);plt.show()
# cords = np.asarray(np.column_stack(np.where(edges > 0)))
# cords = np.fliplr(cords)
# plt.imshow(I);
# plt.scatter(cords[:, 0], cords[:, 1]);plt.show()
#
#
# pca = PCA(n_components=2)
# vals = pca.fit_transform(cords)
# pts = np.matmul(vals, pca.components_) + np.mean(cords, axis=0)
#
# plt.imshow(I);
# plt.scatter(pts[:, 0], pts[:, 1]);plt.show()
# best_ellipse = fit_ellipse.FitEllipse_RANSAC_Support(cords, cv2.cvtColor(I, cv2.COLOR_RGB2GRAY), None, max_itts=50, max_refines=30, max_perc_inliers=99.0)
# best_ellipse = fit_ellipse.FitEllipse_RobustLSQ(cords, cv2.cvtColor(I, cv2.COLOR_RGB2GRAY), None)