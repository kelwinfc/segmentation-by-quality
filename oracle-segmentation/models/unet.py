# Keras
from keras import layers as KL
from keras.initializers import glorot_uniform
from keras import regularizers, optimizers
from keras.layers.core import Reshape
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import copy
import os

if os.sys.version_info[0] < 3:
    from itertools import izip
else:
    izip = zip

from . import BaseNetwork


from keras import backend as K


DICE_SMOOTH = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + DICE_SMOOTH) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + DICE_SMOOTH)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


class UNet(BaseNetwork):
    def build_network(self, X=None, y=None):
        self.conv_padding = 'same'

        shape = (self.img_size[0], self.img_size[1], 3)

        input_layer = KL.Input(shape=shape, name='input')
        last_ = input_layer

        # Convolutional section
        last_conv_per_level = []
        sizes = []
        for conv_level in range(self.conv_num):
            nfilters = self.conv_filter_num * 2 ** conv_level
            sizes.append(nfilters)
            print('nfilters:', nfilters)

            # Convolutional layers
            for c in range(self.conv_consecutive):
                #cfs = max(2, self.conv_filter_size[0] - conv_level)
                #cfs = (cfs, cfs)
                cfs = self.conv_filter_size

                glorot = glorot_uniform(seed=42 + c +
                                        conv_level * self.conv_consecutive)
                last_ = KL.Conv2D(nfilters, cfs,
                                  activation=self.conv_activation,
                                  padding=self.conv_padding,
                                  kernel_initializer=glorot,
                                  name='conv%d-%d' % (conv_level, c))(last_)

            last_conv_per_level.append(last_)

            # Pooling layer
            if conv_level != self.conv_num:
                last_ = KL.MaxPooling2D(pool_size=self.pool_size,
                                        strides=self.pool_strides,
                                        name='pool%d' % conv_level)(last_)

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
        
        out = KL.Conv2D(1, (1, 1), activation='linear')(last_)
        out = KL.Activation('sigmoid', name='output')(out)

        model = Model(inputs=[input_layer], outputs=[out])
        model.compile('adadelta', loss=dice_coef_loss,
                      metrics={'output': dice_coef})

        self.final_shape = out._keras_shape

        return model, model

    def get_generator(self, path, subset):
        self.augment['rescale'] = 1. / 255.
        return super(UNet, self).get_generator(path, subset)

    def transform(self, imgs):
        i_ = imgs / 255.
        preds = self.network.predict(i_)
        return preds
