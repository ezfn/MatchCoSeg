import numpy as np
import numpy.linalg as alg
import math
from lifetobot_sdk.Geometry import coordinate_transformations as geom_trans
class Expander(object):

   def __init__(self, Hhat, src_centroid, src_prncpl_axes, id=None):
       self.was_stopped = False
       self.was_expanded = False
       self.was_rejected = False
       self.Hhat = Hhat
       self.src_centroid = src_centroid
       self.src_prncpl_axes = src_prncpl_axes
       self.id = id
       self.area = math.pi * alg.norm(self.srcPrincipalAxes[:, 0]) * alg.norm(self.srcPrincipalAxes[:, 1]);#ellipse area
       self.currentCorrespondences = dict(src=[],tgt=[],qualities=[],cov_mats=[],inv_cov_mats=[])
       trans_det = alg.det(self.Hhat[0:2,0:2])
       if trans_det > Expander.maximalTransformInflation or \
           trans_det < 1/Expander.maximalTransformInflation \
           or alg.norm(self.src_prncpl_axes[:,1]) < Expander.minimalAxisLen:
           self.was_stopped = True
       else:
           self.H_initial = self.Hhat.copy()
           self.H_prev = self.Hhat.copy()
           self.currentCorrespondences['src'] = np.repeat(self.src_centroid, 3, axis=0) + \
                                                [[0,0],[self.src_prncpl_axes[:,0]],[self.src_prncpl_axes[:,1]]]
           self.currentCorrespondences['dst'] = geom_trans.do_affine_transform(self.currentCorrespondences['src'],
                                                                               self.Hhat)
           self.currentCorrespondences['qualities'] = 0.5*np.ones(3,1) #we don't really know the qualities
           self.currentCorrespondences['cov_mats'] = np.repeat(100*np.eye(2,2)[:,:,None], 3, axis=2)  # we don't really know the cov
           self.currentCorrespondences['inv_cov_mats'] = np.repeat((1/100)*np.eye(2,2)[:,:,None], 3, axis=2)  # we don't really know the cov

           self.invHerrCovariance = []


           repmat(expander.srcPtsCentroid, 3, 1) + [[0 0];
           expander.srcPrincipalAxes
           '];

           pass


   timesSigmaPredictionSlackFactor = 2            #The mahalnobis distance allowed for determining the allowed prediction error
   minimalAreaGrowthFactor = 0.5                  #The minimal allowed rate of a desired expansion factor - when breached, expansion will stop
   minimalSearchWindowExtent = 3                   #minimal side length for the correlation window
   maximalSearchWindowExtent = 16                  #maximal side length for the correlation window
   templateExtent = 8                             #the distance between the template center and any of the sides (square shaped templates)
   correlationDeviationTsh = 0.7                   #How much deviation allowed from the maximal detected correlation
   maximalTransformInflation = 50                 #a boundary on the scaling of a given affine transformation
   minimalInitialSamples = 4              # lower bound of minimal points for initial expansion
   minimalAxisLen = 5
