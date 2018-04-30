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
from .generators import deformed_mask_generator, model_based_mask_generator, \
    combined_generators


DICE_SMOOTH = 1.


def lambda_negative(x):
    return 1 - x


def lambda_diff(x):
    return x[0] - x[1]


def lambda_neg(x):
    return -x


def dice_coef_one(ytrue, ypred):
    ytrue_ = ytrue.flatten()
    ypred_ = ypred.flatten()

    return 2. * (np.sum(ytrue_ * ypred_) + DICE_SMOOTH) / \
        (ytrue_.sum() + ypred_.sum() + DICE_SMOOTH)


class MetricBasedSegmentationNetwork(BaseNetwork):
    def build_network(self):
        img_shape = (self.img_size[0], self.img_size[1], 3)
        mask_shape = (self.img_size[0], self.img_size[1], 1)

        image_input = Input(shape=img_shape, name='image')
        fg_mask_input = Input(shape=mask_shape, name='mask')
        bg_mask_input = Lambda(lambda_negative)(fg_mask_input)

        last_img = [image_input, image_input]
        last_mask = [fg_mask_input, bg_mask_input]

        # Convolutional layers
        for conv_level in range(self.conv_num):
            nfilters = self.conv_filter_num * 2 ** conv_level

            for c in range(self.conv_consecutive):
                last_img, last_mask = \
                    self.gossip_conv_block(last_img, last_mask, nfilters)

            # Pooling layer
            last_img, last_mask = self.gossip_pool_block(last_img, last_mask)

        # Flatten layers
        last_ = last_img
        last_ = [Flatten()(l) for l in last_]

        # Dense layers per stream
        for dense_level in range(self.dense_stream_num):
            if self.shared and False:
                d = Dense(self.dense_stream_width, activation='relu',
                          kernel_regularizer=regularizers.l2(self.l2),
                          kernel_initializer=glorot_uniform(seed=42))
                dense = lambda: d
            else:
                dense = \
                    lambda: Dense(self.dense_stream_width, activation='relu',
                                  kernel_regularizer=regularizers.l2(self.l2),
                                  kernel_initializer=glorot_uniform(seed=42))

            last_ = [dense()(l) for l in last_]

        #norm = Lambda(lambda x: x / (K.sqrt(K.sum(K.square(x)) + 1e-10)))
        #last_ = [norm(l) for l in last_]

        # Concatenation
        last_ = Concatenate()(last_)
        #last_ = Multiply()(last_)

        # Dense layers
        for dense_level in range(self.dense_num):
            last_ = Dense(self.dense_width, activation='relu',
                          kernel_regularizer=regularizers.l2(self.l2),
                          kernel_initializer=glorot_uniform(seed=42)
                          )(last_)

            if self.dropout > 0.:
                last_ = Dropout(self.dropout, seed=42)(last_)

        last_ = Dense(1, activation='linear',
                      kernel_initializer=glorot_uniform(seed=42)
                      )(last_)

        output = Activation('linear', name='output')(last_)
        opt_output = Activation('sigmoid', name='output')(last_)

        opt_model = Model(inputs=[image_input, fg_mask_input],
                          outputs=[opt_output])
        opt_model.compile(optimizers.Adam(lr=1e-4),
                      #'adadelta',
                      loss='mse',
                      metrics=['mae'])

        model = Model(inputs=[image_input, fg_mask_input], outputs=[output])
        model.compile(optimizers.Adam(lr=1e-4),
                      #'adadelta',
                      loss='mse',
                      metrics=['mae'])

        return opt_model, opt_model

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
        act_diff = [Lambda(lambda_diff)([this_, other_])
                    for this_, other_ in zip(images, images[::-1])]

        #this_high = [Activation('relu')(d) for d in act_diff]

        other_high = [Activation('relu')(Lambda(lambda_neg)(d)) for d in act_diff]
        
        # FIXME: without relu
        #other_high = [Lambda(lambda_neg)(d) for d in act_diff]

        stimuli = [Lambda(lambda_neg)(Multiply()([m, d]))
                   for m, d in zip(masks, other_high)]

        """
        stimuli = [Lambda(lambda_diff)([p, n])
                   for p, n in zip(pos_stimuli, neg_stimuli)]
        #stimuli = Lambda(lambda_diff)(stimuli)
        #stimuli = [stimuli, Lambda(lambda_neg)(stimuli)]
        """

        #stimuli = [Lambda(lambda x: self.reciprocal_stimuli * x)(s)
                   #for s in stimuli]

        # Second merge
        images = [Add()([i, s]) for i, s in zip(images, stimuli)]
        #images = [Activation('relu')(i) for i in images]

        return images, masks

    def gossip_pool_block(self, images, masks):
        pool = lambda: AveragePooling2D(pool_size=self.pool_size,
                                        strides=self.pool_strides)
        images = [pool()(l) for l in images]
        masks = [pool()(l) for l in masks]
        return images, masks

    def fit_from_dir(self, path):
        def isubpath(subset):
            #FIXME
            return os.path.join(path, 'test', subset, 'seg')
            return os.path.join(path, 'validation', subset, 'seg')

        def ipath(subset, i):
            return os.path.join(isubpath(subset), i)

        super(MetricBasedSegmentationNetwork, self).fit_from_dir(path)

        imgs = sorted(list(os.walk(isubpath('imgs')))[0][2])
        masks = sorted(list(os.walk(isubpath('masks')))[0][2])

        imgs = [self.preprocess(cv2.imread(ipath('imgs', i))[:, :, ::-1])
                for i in imgs]
        masks = [self.preprocess(cv2.imread(ipath('masks', i),
                                            0))[:, :, np.newaxis]
                 for i in masks]

        grad_masks = [self.get_base_mask(255 * i) for i in imgs]
        orig_grad_masks = [m.copy() for m in grad_masks]

        #loss = mean_squared_error(K.constant([[1]]),
                                 #self.opt_network.get_layer('output').output)
        neg_loss = self.opt_network.get_layer('output').output
        grads = K.gradients(neg_loss, self.opt_network.get_layer('mask').input)
        grads_fn = K.function([self.opt_network.get_layer('image').input,
                               self.opt_network.get_layer('mask').input], grads)
        performance = np.zeros((self.max_grad_iterations, len(imgs)))
        round_performance = np.zeros((self.max_grad_iterations, len(imgs)))

        best_iter = 0
        best_performance = np.mean([dice_coef_one(gt_mask, grad_mask.round()) for gt_mask, grad_mask in zip(masks, grad_masks)])
        print(best_iter, best_performance)

        history = [best_performance]
        
        #eta = 5 * 1e-2
        #eta = 1 * 1e-1
        eta = 0.1

        for iter_ in range(self.max_grad_iterations):
            for idx_, (img, gt_mask, grad_mask, orig_grad_mask) in \
                    enumerate(zip(imgs, masks, grad_masks, orig_grad_masks)):

                real_grads, = grads_fn([img[np.newaxis],
                                        grad_mask[np.newaxis]])

                real_grads /= np.abs(real_grads).max() + 1e-15
                real_grads = 2. / (1. + np.exp(-5.*real_grads)) - 1.

                #grad_mask += eta / (1 + np.log(iter_ + 1)) * real_grads[0]
                #grad_mask += eta / np.sqrt(iter_ + 1) * real_grads[0]
                #grad_mask += eta * (0.9 ** iter_) * real_grads[0]
                grad_mask += eta * real_grads[0]
                #print(np.sort(grad_mask.ravel()))
                grad_mask[grad_mask < 0] = 0
                grad_mask[grad_mask > 1] = 1

                #FIXME

                grad_masks[idx_] = grad_mask

                if idx_ == 0:
                    cv2.imwrite('img-%04d.png' % iter_,
                                (np.hstack((img, cv2.merge(3 * [grad_mask[:, :, 0].round()]))) * 255).astype(np.uint8)[:, :, ::-1])

                performance[iter_][idx_] = dice_coef_one(gt_mask, grad_mask)
                round_performance[iter_][idx_] = dice_coef_one(gt_mask,
                                                               grad_mask.round())

            #print(np.sort(performance[iter_, :]))
            next_performance = np.mean(round_performance[iter_, :])

            improved = False
            if next_performance > best_performance:
                best_performance = next_performance
                best_iter = iter_ + 1
                improved = True

            history.append(next_performance)

            print(' '.join(list(map(str, [iter_ + 1, next_performance, np.mean(performance[iter_, :]), '*' if improved else '']))))

        self.grad_iterations = best_iter
        print('Back iterations:', self.grad_iterations)
        print(history)

        return self

    def preprocess(self, img):
        img = cv2.resize(img, self.img_size[: 2]).astype(np.float)
        img = img / 255.
        return img

    def transform(self, imgs):
        self.img_shape = (128, 128)
        return [self.img_transform(img) for img in imgs]

    def predict_quality(self, imgs, masks):
        preproc_imgs = np.asarray([self.preprocess(i) for i in imgs])
        preproc_masks = np.asarray([self.preprocess(m)[:, :, np.newaxis]
                                    for m in masks])
        
        ret = self.network.predict({'image': preproc_imgs,
                                    'mask': preproc_masks})[:, 0]
        return ret

    def img_transform(self, img):
        orig_size = img.shape[: 2][::-1]

        img_ = cv2.resize(img.astype(np.float32), self.img_shape)
        mask = self.get_base_mask(img_)

        img_ /= 255.

        #loss = mean_squared_error(K.constant([[1]]),
        #                          self.network.get_layer('output').output)
        neg_loss = self.opt_network.get_layer('output').output
        grads = K.gradients(neg_loss, self.opt_network.get_layer('mask').input)

        grads_fn = K.function([self.opt_network.get_layer('image').input,
                               self.opt_network.get_layer('mask').input],
                              grads)

        eta = 0.1#5 * 1e-2

        prev_val = -np.inf

        orig_mask = mask.copy()

        for i in range(self.grad_iterations):
            real_grads, = grads_fn([img_[np.newaxis], mask[np.newaxis]])

            real_grads /= np.abs(real_grads).max() + 1e-15
            real_grads = 2. / (1. + np.exp(-5.*real_grads)) - 1.

            #if real_grads.min() == 0 and real_grads.max() == 0:
                #print('Cannot optimize more after %d iterations' % i)
                #break

            #if np.any(real_grads != 0):
                #try:
                    #pos_norm = real_grads[real_grads > 0].max()
                #except:
                    #pos_norm = 1e-15

                #try:
                    #neg_norm = -real_grads[real_grads < 0].min()
                #except:
                    #neg_norm = 1e-15

                #pos_norm = max(pos_norm, neg_norm)
                #neg_norm = max(pos_norm, neg_norm)

                #real_grads[real_grads > 0] /= pos_norm
                #real_grads[real_grads < 0] /= neg_norm

            #mask += eta / sqrt(i + 1) * real_grads[0]
            mask += eta * real_grads[0]

            mask[mask < 0] = 0
            mask[mask > 1] = 1

            #FIXME
            #mask[(mask > 0.5) != (orig_mask > 0.5)] = np.round(mask[(mask > 0.5) != (orig_mask > 0.5)])
            #mask = np.round(mask)

            #cv2.imwrite('img-%04d.png' % i,
                        #(np.hstack((img, cv2.merge(3 * [mask[:, :, 0].round()]))) * 255).astype(np.uint8))

        mask = mask.round()

        #cv2.imshow('img', np.hstack((img, cv2.merge(3 * [mask]) * 255)).astype(np.uint8))
        #cv2.waitKey(0)
        mask = cv2.resize(mask, orig_size)

        return mask

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
            #print(best_model)
            #out_mask = masks[best_model]

            return masks[best_model]

    #def get_generator(self, path, subset):
        #ret = self.get_base_generator(path, subset)
        #return deformed_mask_generator(ret, os.path.basename(path))

    def get_generator(self, path, subset):
        ret1 = self.get_base_generator(path, subset)
        ret2 = self.get_base_generator(path, subset)
        g1 = model_based_mask_generator(self.base_models, ret1)
        g2 = deformed_mask_generator(ret2, os.path.basename(path))

        if len(self.base_models) > 0:
            return g1

        print('no base models')
        return g2
        return combined_generators([g1, g2], seed=42)


class ModelSelectionSegmentationNetwork(MetricBasedSegmentationNetwork):
    def transform(self, imgs):
        self.grad_iterations = 0
        self.img_shape = (128, 128)
        return super(ModelSelectionSegmentationNetwork, self).transform(imgs)

    def fit_from_dir(self, path):
        self.max_grad_iterations = 0
        super(ModelSelectionSegmentationNetwork, self).fit_from_dir(path)
        return self


class SingleStreamMetricSegmentationNetwork(MetricBasedSegmentationNetwork):
    def build_network(self):
        img_shape = (self.img_size[0], self.img_size[1], 3)
        mask_shape = (self.img_size[0], self.img_size[1], 1)

        image_input = Input(shape=img_shape, name='image')
        mask_input = Input(shape=mask_shape, name='mask')
        
        last_ = Concatenate()([image_input, mask_input])
        
        # Convolutional layers
        for conv_level in range(self.conv_num):
            nfilters = self.conv_filter_num * 2 ** conv_level
            
            for c in range(self.conv_consecutive):
                cfs = self.conv_filter_size
                glorot = glorot_uniform(seed=42)

                last_ = Conv2D(nfilters, cfs,
                               activation=self.conv_activation,
                               padding=self.conv_padding,
                               kernel_initializer=glorot)(last_)

            # Pooling layer
            last_ = AveragePooling2D(pool_size=self.pool_size,
                                     strides=self.pool_strides)(last_)

        # Flatten layers
        last_ = Flatten()(last_)

        # Dense layers
        for dense_level in range(self.dense_num):
            last_ = Dense(self.dense_width, activation='relu',
                          kernel_initializer=glorot_uniform(seed=42)
                          )(last_)

            if self.dropout > 0.:
                last_ = Dropout(self.dropout, seed=42)(last_)

        last_ = Dense(1, activation='linear',
                      kernel_initializer=glorot_uniform(seed=42)
                      )(last_)
        output = Activation('sigmoid', name='output')(last_)

        model = Model(inputs=[image_input, mask_input], outputs=[output])
        model.compile(optimizers.Adam(lr=1e-4),
                      #'adadelta',
                      loss='mae',
                      metrics=['mae'])

        return model, model


class DualStreamMetricSegmentationNetwork(MetricBasedSegmentationNetwork):
    def build_network(self):
        img_shape = (self.img_size[0], self.img_size[1], 3)
        mask_shape = (self.img_size[0], self.img_size[1], 1)

        image_input = Input(shape=img_shape, name='image')
        mask_input = Input(shape=mask_shape, name='mask')

        last_ = [image_input, mask_input]

        # Convolutional layers
        for conv_level in range(self.conv_num):
            nfilters = self.conv_filter_num * 2 ** conv_level
            
            for c in range(self.conv_consecutive):
                cfs = self.conv_filter_size
                glorot = glorot_uniform(seed=42)

                last_ = [Conv2D(nfilters, cfs,
                                activation=self.conv_activation,
                                padding=self.conv_padding,
                                kernel_initializer=glorot)(l)
                         for l in last_]

            # Pooling layer
            last_ = [AveragePooling2D(pool_size=self.pool_size,
                                      strides=self.pool_strides)(l)
                     for l in last_]

        # Flatten layers
        last_ = [Flatten()(l) for l in last_]

        # Dense layers per stream
        for dense_level in range(self.dense_stream_num):
            dense = lambda: Dense(self.dense_stream_width, activation='relu',
                                  kernel_initializer=glorot_uniform(seed=42))

            last_ = [dense()(l) for l in last_]

        # Concatenation
        last_ = Concatenate()(last_)

        # Dense layers
        for dense_level in range(self.dense_num):
            last_ = Dense(self.dense_width, activation='relu',
                          kernel_initializer=glorot_uniform(seed=42)
                          )(last_)

            if self.dropout > 0.:
                last_ = Dropout(self.dropout, seed=42)(last_)

        last_ = Dense(1, activation='linear',
                      kernel_initializer=glorot_uniform(seed=42)
                      )(last_)
        output = Activation('linear', name='output')(last_)

        model = Model(inputs=[image_input, mask_input], outputs=[output])
        model.compile(optimizers.Adam(lr=1e-4),
                      #'adadelta',
                      loss='mse',
                      metrics=['mae'])

        return model, model
