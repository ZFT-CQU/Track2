import numpy as np
from skimage.measure import regionprops
import cv2

CORRECTION = 255


def nodule_size(nodule):
    tmp = nodule.copy()
    tmp[tmp != 0] = 255
    tmp = tmp.astype(int)
    properties = regionprops(tmp)
    width = 0.0
    height = 0.0
    for p in properties:
        min_row, min_col, max_row, max_col = p.bbox
        width = max_row - min_row
        height = max_col - min_col
    return width, height


def convert_to_range_0_1(image_data):
    """
    Normalize image to be between 0 and 1
        image_data: the image to normalize
    returns the normalized image
    """
    image_max = max(image_data.flatten())
    image_min = min(image_data.flatten())
    try:
        return (image_data - image_min) / (image_max - image_min)
    except Exception:
        print('invalid value encounteered')
        return image_data


def contrast_matching(nodule_2d, lung_photo):
    """
     Contrast matching according to Litjens et al.
     With some additional clip to prevent negative values or 0.
      nodule_2d: intensities of the nodule
      lung_photo: intensities of this particular lung area
     returns c, but is clipped to 0.4 since low values made the nodules neigh
     invisible sometimes.
    """
    # mean from only nodule pixels
    indexes = nodule_2d != np.min(nodule_2d)
    it = np.mean(nodule_2d[indexes].flatten())

    # mean of the surrounding lung tissue
    ib = np.mean(lung_photo.flatten())

    # determine contrast
    c = np.log(it / ib)

    return max(0.3, c)


def poisson_blend(nodule, lung_photo, x0, x1, y0, y1):

    im = lung_photo
    center = (int(np.round((x1 + x0) / 2)), int(np.round((y1 + y0) / 2)))

    # determine the smallest box that can be drawn around the nodule pixels
    non_zero = np.argwhere(nodule)
    top_left = non_zero.min(axis=0)
    bottom_right = non_zero.max(axis=0)
    nodule = nodule[top_left[0]:bottom_right[0] + 1,
                    top_left[1]:bottom_right[1] + 1]

    obj = nodule

    # convert np to cv2
    cv2.imwrite('test_img.jpg', im * 255)
    cv2.imwrite('test_obj.jpg', obj * 255)
    im2 = cv2.imread('test_img.jpg')
    obj2 = cv2.imread('test_obj.jpg')

    # add gaussian blurring to reduce artefacts
    cv2.imwrite('test_obj_masked2.jpg', obj2 / 255)

    # apply correction mask
    mask2 = np.ones(obj2.shape, obj2.dtype) * CORRECTION

    # Poisson blend the images
    mixed_clone2 = cv2.seamlessClone(obj2, im2, mask2, center,
                                     cv2.MIXED_CLONE)
    cv2.imwrite('diseased.jpg', mixed_clone2)
    return cv2.cvtColor(mixed_clone2, cv2.COLOR_BGR2GRAY)
