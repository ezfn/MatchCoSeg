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

root_dir = '/home/rd/Downloads'
train_dir = os.path.join(root_dir, 'images/train')
test_dir = os.path.join(root_dir, 'images/test')
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





for label_idx, label in enumerate(test_labels):
    Iout = np.zeros((50, 50), dtype=np.uint8)
    is_ellipse = os.path.join(label[0].split(' ')[1])
    if is_ellipse == 'True':
        ellipse_params = np.array(list(map(float, label[1:])))
        # I = cv2.imread(os.path.join(root_dir, label[0].split(' ')[0]))
        cv2.ellipse(img=Iout, center=(int(ellipse_params[0]), int(ellipse_params[1])),
                    axes=(int(ellipse_params[3]), int(ellipse_params[4])), angle=int(ellipse_params[2]),
                    startAngle=0, endAngle=360, color=(255, 255, 255), thickness=1)
        Iout = cv2.blur(Iout, (3, 3))
    else:
        pass

    Iout_file = os.path.join(root_dir, label[0].split(' ')[0]).replace('.jpg', '_mask.png')
    cv2.imwrite(Iout_file, Iout)
    print('finished {}/{} test labels'.format(label_idx, len(test_labels)))

for label_idx, label in enumerate(train_labels):
    Iout = np.zeros((50, 50), dtype=np.uint8)
    is_ellipse = os.path.join(label[0].split(' ')[1])
    if is_ellipse == 'True':
        ellipse_params = np.array(list(map(float, label[1:])))
        # I = cv2.imread(os.path.join(root_dir, label[0].split(' ')[0]))
        cv2.ellipse(img=Iout, center=(int(ellipse_params[0]), int(ellipse_params[1])),
                    axes=(int(ellipse_params[3]), int(ellipse_params[4])), angle=int(ellipse_params[2]),
                    startAngle=0, endAngle=360, color=(255, 255, 255), thickness=1)
        Iout = cv2.blur(Iout, (3, 3))
    else:
        pass

    Iout_file = os.path.join(root_dir, label[0].split(' ')[0]).replace('.jpg', '_mask.png')
    cv2.imwrite(Iout_file, Iout)
    print('finished {}/{} train labels'.format(label_idx, len(test_labels)))

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