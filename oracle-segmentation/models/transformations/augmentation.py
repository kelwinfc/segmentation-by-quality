"""
Extends Keras image data generation with our contrast stretching transformation.
"""

import numpy as np
import re
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from skimage.exposure import adjust_sigmoid
from keras.preprocessing.image import ImageDataGenerator


class MyImageDataGenerator(ImageDataGenerator):
    def __init__(self, contrast_stretching_perc=0., **ARGS):
        self.contrast_stretching_perc = contrast_stretching_perc
        super(MyImageDataGenerator, self).__init__(**ARGS)

    def random_transform(self, x, seed=None):
        x = super(MyImageDataGenerator, self).random_transform(x, seed)
        if self.contrast_stretching_perc != 0.:
            contrast_p = np.random.uniform(1. - self.contrast_stretching_perc,
                                           1. + self.contrast_stretching_perc)
            x = adjust_sigmoid(x.astype(np.float) / 255., cutoff=0.5,
                               gain=5 * contrast_p, inv=False)
            x *= 255
        return x
