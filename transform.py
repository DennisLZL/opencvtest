import numpy as np
import cv2

def order_points(pts):
    rect = np.zeros((4, 2), dtype = 'float32')
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
def four_point_transform(imag, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br - bl) ** 2).sum())
    widthB = np.sqrt(((tr - tl) ** 2).sum())
    heightA = np.sqrt(((bl - tl) ** 2).sum())
    heightB = np.sqrt(((br - tr) ** 2).sum())
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0], 
        [maxWidth - 1, 0], 
        [maxWidth - 1, maxHeight - 1], 
        [0, maxHeight - 1]], dtype = 'float32')

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(imag, M, (maxWidth, maxHeight))
    
    return warped


