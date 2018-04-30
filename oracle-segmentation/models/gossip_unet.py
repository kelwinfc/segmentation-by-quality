from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt
import tensorflow as tf
import numpy as np
import pickle
import cv2

# Keras
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, UpSampling2D, \
    Conv2DTranspose, SeparableConv2D, Conv3D, Cropping2D
from keras.layers import Input, Dense, Dropout, Flatten, ThresholdedReLU, \
    LeakyReLU
from keras import layers as KL
from keras.losses import mean_squared_error
from keras.layers.core import Activation, Lambda, ActivityRegularization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.initializers import glorot_uniform, Zeros, Constant
from keras.layers.merge import Concatenate, Maximum, Add, Multiply
from keras import regularizers, optimizers
from keras.utils import to_categorical
from keras.layers.core import Reshape
from keras import backend as K
from keras.models import Model
import os


from . import BaseNetwork
from .generators import deformed_mask2mask_generator


DICE_SMOOTH = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = K.minimum(1., y_pred_f)
    y_pred_f = K.maximum(0., y_pred_f)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + DICE_SMOOTH) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + DICE_SMOOTH)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def corrected_dice(y_true, y_pred):
    print(y_pred.shape)
    return 0.


class GossipUNet(BaseNetwork):
    def build_network(self):
        self.conv_padding = 'same'

        shape = (self.img_size[0], self.img_size[1], 3)
        self.img_shape = shape
        self.mask_shape = (self.img_size[0], self.img_size[1], 1)

        image_input = Input(shape=self.img_shape, name='image')
        fg_mask_input = Input(shape=self.mask_shape, name='mask')
        bg_mask_input = Lambda(lambda x: 1 - x)(fg_mask_input)

        last_img = [image_input, image_input]
        last_mask = [fg_mask_input, bg_mask_input]

        self.subtract = Lambda(lambda x: x[0] - x[1], name='subtract')
        self.relu = LeakyReLU(0.25)
        #self.relu = Activation('relu')
        self.neg = Lambda(lambda x: -x)
        self.add = Add()
        self.mul = Multiply()

        #last_ = input_layer

        # Convolutional section
        last_conv_per_level = []
        sizes = []
        for conv_level in range(self.conv_num + 1):
            nfilters = self.conv_filter_num * 2 ** conv_level
            sizes.append(nfilters)

            # Convolutional layers
            for c in range(self.conv_consecutive):
                cfs = self.conv_filter_size

                last_img, last_mask = \
                    self.gossip_conv_block(last_img, last_mask, nfilters)

            last_conv_per_level.append(KL.Concatenate(axis=3)(last_img +
                                                              last_mask
                                                              ))

            # Pooling layer
            if conv_level != self.conv_num:
                last_img, last_mask = \
                    self.gossip_pool_block(last_img, last_mask)

        last_ = KL.Concatenate(axis=3)(last_img + last_mask)

        # Deconvolutional section
        for conv_level in range(self.conv_num)[::-1]:
            cc = KL.Concatenate(axis=3)

            last_ = cc(
                    [KL.Conv2DTranspose(sizes[conv_level],
                                        self.pool_size,
                                        strides=(self.pool_size,
                                                 self.pool_size))(last_),
                    last_conv_per_level[conv_level]])

            for c in range(self.conv_consecutive):
                cfs = self.conv_filter_size

                last_ = KL.Conv2D(sizes[conv_level], cfs,
                                  activation=self.conv_activation,
                                  kernel_initializer=glorot_uniform(seed=42),
                                  padding=self.conv_padding)(last_)

        output = KL.Conv2D(1, (1, 1), activation='linear')(last_)
        #output = KL.Activation('sigmoid', name='output')(output)
        output = KL.Activation('tanh', name='output')(output)

        corrected_output = KL.Add(name='corrected-output')([fg_mask_input,
                                                            output])

        model = Model(inputs=[image_input, fg_mask_input],
                      outputs=[output, corrected_output])
        model.compile('adadelta',
                      loss={'output': 'mse',
                            'corrected-output': dice_coef_loss
                            },
                      loss_weights={'output': 1.,
                                    'corrected-output': 0.},
                      metrics={#'output': 'mae',
                               'corrected-output': dice_coef
                               })

        self.final_shape = output._keras_shape

        return model, model

    def gossip_conv_block(self, images, masks, nfilters):
        cfs = self.conv_filter_size

        conv = lambda: Conv2D(nfilters, cfs, activation='linear',
                              padding=self.conv_padding,
                              kernel_initializer=glorot_uniform(seed=42))
        if self.shared:
            cc = conv()
            conv = lambda: cc
        images = [conv()(l) for l in images]

        """
        constant = Constant(value=1. / float(cfs[0] * cfs[1]))
        mask_conv = Conv2D(1, cfs, activation='linear',
                            padding=self.conv_padding,
                            kernel_initializer=constant,
                            bias_initializer=Zeros(),
                            trainable=False)
        mask_conv.trainable = False

        masks = [mask_conv(l) for l in masks]
        """

        if self.conv_padding == 'valid':
            crop_cfs = int(floor(cfs[0] / 2))
            crop_cfs = (crop_cfs, crop_cfs)
            crop = Cropping2D(cropping=(crop_cfs, crop_cfs))
            masks = [crop(m) for m in masks]

        # Merge connections
        act_diff = [self.subtract([this_, other_])
                    for this_, other_ in zip(images, images[::-1])]

        #this_high = [self.relu(d) for d in act_diff]

        other_high = [self.relu(self.neg(d)) for d in act_diff]
        
        # FIXME: without relu
        #other_high = [self.neg(d) for d in act_diff]

        stimuli = [self.neg(self.mul([m, d]))
                   for m, d in zip(masks, other_high)]

        """
        stimuli = [self.subtract([p, n])
                   for p, n in zip(pos_stimuli, neg_stimuli)]
        #stimuli = self.subtract(stimuli)
        #stimuli = [stimuli, self.neg(stimuli)]
        """

        #stimuli = [Lambda(lambda x: self.reciprocal_stimuli * x)(s)
                   #for s in stimuli]

        # Second merge
        images = [self.add([i, s]) for i, s in zip(images, stimuli)]
        images = [self.relu(i) for i in images]

        return images, masks

    def gossip_pool_block(self, images, masks):
        pool = lambda: MaxPooling2D(pool_size=self.pool_size,
                                    strides=self.pool_strides)
        #pool = lambda: AveragePooling2D(pool_size=self.pool_size,
                                       #strides=self.pool_strides)
        images = [pool()(l) for l in images]
        masks = [pool()(l) for l in masks]
        return images, masks

    def fit_from_dir(self, path):
        def isubpath(subset):
            return os.path.join(path, 'validation', subset, 'seg')

        def ipath(subset, i):
            return os.path.join(isubpath(subset), i)

        super(GossipUNet, self).fit_from_dir(path)

        return self

    def preprocess(self, img):
        img = cv2.resize(img, self.img_size[: 2]).astype(np.float)
        img = img / 255.
        return img

    def transform(self, imgs):
        self.img_shape = (128, 128)
        return [self.img_transform(img) for img in imgs]

    def img_transform(self, img):
        img_ = img.astype(np.float) / 255.
        mask = self.get_base_mask(img_)
        for _ in range(100):
            diff = self.network.predict({'image': img_[np.newaxis],
                                         'mask': mask[np.newaxis]})[1][0]
            mask = diff
            #mask += diff * 0.25
            mask[mask > 1] = 1
            mask[mask < 0] = 0

        mask = mask.round()
        cv2.imwrite('mask.png', np.hstack(((img, cv2.merge(3 * [(255 * mask).astype(np.uint8)])))))

        return mask[:, :, 0]

    def get_base_mask(self, img):
        if len(self.base_models) == 0 or self.from_zero:
            #return np.ones((img.shape[0], img.shape[1], 1))
            return np.zeros((img.shape[0], img.shape[1], 1))
        else:
            masks = []
            images = []

            for mid in np.arange(len(self.base_models)):
                with tf.variable_scope(self.base_models[mid][0]):
                    next_mask = self.base_models[mid][1].transform(img[np.newaxis])
                    masks.append(next_mask[0])
                    images.append(img / 255.)

            scores = self.network.predict({'image': np.asarray(images),
                                        'mask': np.asarray(masks)})

            best_model = np.argmax(scores)
            print(best_model)
            out_mask = masks[best_model]

            return masks[best_model]

    def get_generator(self, path, subset):
        ret = self.get_base_generator(path, subset)
        return deformed_mask2mask_generator(ret)

    """
    def get_generator(self, path, subset):
        ret1 = self.get_base_generator(path, subset)
        ret2 = self.get_base_generator(path, subset)
        g1 = model_based_mask_generator(self.base_models, ret1)
        g2 = deformed_mask_generator(ret2)

        return deformed_mask_generator(ret1)
        return combined_generators([g1, g2], seed=42)
    """



"""
    def get_generator(self, path, subset):
        self.augment['rescale'] = 1. / 255.
        return super().get_generator(path, subset)

    def transform(self, imgs):
        i_ = imgs / 255.
        preds = self.network.predict(i_)
        return preds
"""