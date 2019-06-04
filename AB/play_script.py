import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from AB import fit_ellipse
import importlib
importlib.reload(fit_ellipse)

I = cv2.imread('/media/Media/MEDIA/datasets/images/train/0415.jpg')
plt.imshow(I);plt.show()
edges = cv2.Canny(I,100,200)
plt.imshow(edges);plt.show()
cords = np.asarray(np.column_stack(np.where(edges > 0)))
cords = np.fliplr(cords)
plt.imshow(I);
plt.scatter(cords[:, 0], cords[:, 1]);plt.show()


pca = PCA(n_components=2)
vals = pca.fit_transform(cords)
pts = np.matmul(vals, pca.components_) + np.mean(cords, axis=0)

plt.imshow(I);
plt.scatter(pts[:, 0], pts[:, 1]);plt.show()
best_ellipse = fit_ellipse.FitEllipse_RANSAC_Support(cords, cv2.cvtColor(I, cv2.COLOR_RGB2GRAY), None, max_itts=50, max_refines=30, max_perc_inliers=99.0)
best_ellipse = fit_ellipse.FitEllipse_RobustLSQ(cords, cv2.cvtColor(I, cv2.COLOR_RGB2GRAY), None)