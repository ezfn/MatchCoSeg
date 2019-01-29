import cv2
import numpy as np

def is_odd(num):
    return num & 0x1

def patched_filter(tgtImg, template, corr_tsh):
    result_dict = dict(is_garbage=False, cords= (np.array(tgtImg.shape[::-1])/2).astype(np.int),
                       max_corr=0, errorCorrMat=np.inf*np.matrix(np.ones((2,2))),initialCorrelation = 0)
    QE = 0.75
    #TODO: check here if the patch is too smooth
    # rejecting homogeneous patch for lack of information
    if (np.sum(np.abs(template - np.mean(template))) <= (3 * template.size)):
        result_dict['is_garbage'] = True
        # correlationResponse = [];
        return result_dict
    templateSize = template.shape
    isOdd = np.array([is_odd(templateSize[0]), is_odd(templateSize[1])])
    templateExtent = (templateSize - isOdd) / 2
    #TODO: cuda this
    C = cv2.matchTemplate(tgtImg.astype(np.float32) , template.astype(np.float32), method=cv2.TM_CCOEFF_NORMED)
    responseSize = C.shape
    result_dict['initialCorrelation'] = C[int(round(responseSize[0] / 2)), int(round(responseSize[1] / 2))]
    max_arg = np.argmax(C)
    max_cord = np.array(np.unravel_index(max_arg,C.shape))
    max_corr = C[max_cord[0],max_cord[1]]
    if (max_corr < corr_tsh):  # rejecting correlation results where there is no value over the minimal value that was set by the user
        result_dict['is_garbage'] = True
        return result_dict

    result_dict['max_corr'] = max_corr
    XX,YY = np.meshgrid(range(0,responseSize[1]),range(0,responseSize[0]))
    XX = XX[C < max_corr * QE]
    YY = YY[C < max_corr * QE]
    C = C[C < max_corr * QE]
    exp_C = np.exp(C)
    C = exp_C / np.sum(exp_C)  # we treat the correlation result as a distribution so we could analyze it statistically
    muX = np.inner(C,XX)
    muY = np.inner(C,YY)
    XX = XX - muX
    YY = YY - muY
    varX = max(np.inner(C,XX**2), 0.25)
    varY = max(np.inner(C,YY**2), 0.25)

    covXY = np.inner(C,XX*YY)
    result_dict['errorCorrMat'] = np.matrix([[varX, covXY],[covXY, varY]])
    result_dict['cords'] = max_cord + (templateExtent - isOdd)[::-1]
    if np.linalg.det(result_dict['errorCorrMat']) > max((np.prod(responseSize,0) / 2), 36):
        result_dict['is_garbage'] = True
    return result_dict
    # cv2.imshow('C',(C*255 * (C > 0)).astype(np.uint8));cv2.waitKey(0)

