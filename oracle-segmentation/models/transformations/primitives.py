import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
from skimage import morphology
from skimage.transform import rotate

def _elastic_transform(image, alpha, sigma, alpha_affine, random_state):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0]+square_size, center_square[1]-square_size],
        center_square - square_size])
    pts2 = pts1 + random_state.uniform(
        -alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))#, np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))#, np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def morphological_transform(mask, size, random_state):
    elem = morphology.disk(abs(size))
    op = morphology.dilation if size > 0 else morphology.erosion
    return op(mask, elem)


def elastic_transform(mask, alpha, sigma, alpha_affine, random_state):
    s = mask.shape[1]
    return _elastic_transform(mask, s*alpha, s*sigma, s*alpha_affine, random_state)


def flip_transform(mask, flipdir, random_state):
    assert flipdir in (0, 1, 2)
    dst = mask.copy()
    if flipdir == 0 or flipdir == 2:
        dst = dst[:, ::-1]
    if flipdir == 1 or flipdir == 2:
        dst = dst[::-1, :]
    return dst


def switch_transform(mask, n, random_state):
    # randomly switch pixels
    ix = random_state.choice(np.prod(mask.shape), n, False)
    x = ix // len(mask)
    y = ix % len(mask)
    dst = mask.copy()
    dst[y, x] = 1-dst[y, x]
    return dst


def rotate_transform(mask, angle, random_state):  # angle in degrees
    return rotate(mask, angle)


def offset_transform(mask, xoffset, yoffset, random_state):
    if xoffset:
        off = np.zeros((mask.shape[0], abs(xoffset)), mask.dtype)
        if xoffset < 0:
            mask = np.c_[off, mask[:,:xoffset]]
        else:
            mask = np.c_[mask[:,xoffset:], off]
    if yoffset:
        off = np.zeros((abs(yoffset), mask.shape[1]), mask.dtype)
        if yoffset < 0:
            mask = np.r_[off, mask[:yoffset]]
        else:
            mask = np.r_[mask[yoffset:], off]
    return mask


def flip_offset_transform(mask, xoffset, yoffset, flipdir, random_state):
    mask = offset_transform(mask, xoffset, yoffset, random_state)
    mask = flip_transform(mask, flipdir, random_state)
    return mask


if __name__ == '__main__':
    from skimage.io import imread, imsave
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    import os
    import sys

    IMG_SHAPE = (128, 128)
    filename = sys.argv[1]

    mask = imread(filename)
    mask = resize(mask, IMG_SHAPE, mode='reflect')

    transformations = [
        ('original', mask),
        ('dilation', morphological_transform(mask, 15, np.random)),
        ('erosion', morphological_transform(mask, -10, np.random)),
        ('elastic', elastic_transform(mask, 3, 0.07, 0.09, np.random)),
        ('flip_offset', flip_offset_transform(mask, 20, -10, 2, np.random)),
        ('switch', switch_transform(mask, 1200, np.random)),
        ('rotate', rotate_transform(mask, 70, np.random)),
    ]

    for name, mask in transformations:
        imsave(filename[:-4] + '-' + name + '.png', mask)
