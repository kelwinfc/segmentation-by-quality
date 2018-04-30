from keras.models import load_model, Sequential
from keras.losses import mean_squared_error
from keras.layers import Lambda
import keras.backend as K
from skimage.transform import resize
from skimage.io import imread
import tensorflow as tf
import numpy as np
import pickle
import sys

from keras.losses import binary_crossentropy
import collections
import numpy as np
import random
import copy
import cv2
import os

# Keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from models.transformations.augmentation import MyImageDataGenerator
from keras.layers import advanced_activations as adactivations
from keras import backend as K


from sklearn.base import BaseEstimator, TransformerMixin


if os.sys.version_info[0] < 3:
    from itertools import izip
else:
    izip = zip

from . import utils


DICE_SMOOTH = 1.
CROSSENTROPY_EPS = 1e-10

def out_accuracy(y_true, y_pred):
    def acc(y, p):
        return np.mean(y == p)

    return np.asarray([acc(y.ravel(), p.ravel())
                       for y, p in zip(y_true, y_pred)])


def out_dice_coef(y_true, y_pred):
    return np.asarray([2. * (np.sum(gt.ravel() * p.ravel()) + DICE_SMOOTH) /
                       (np.sum(gt.ravel()) + np.sum(p.ravel()) + DICE_SMOOTH)
                       for gt, p in zip(y_true, y_pred)])


class BaseNetwork(BaseEstimator, TransformerMixin):
    def __init__(self,
                 conv_num=4,
                 conv_filter_num=32,
                 conv_filter_size=3,
                 conv_activation='relu',
                 conv_padding='valid',
                 conv_consecutive=2,
                 reciprocal_stimuli=0.5,
                 pool_size=2,
                 pool_strides=None,
                 dense_stream_num=1,
                 dense_stream_width=512,
                 dense_num=1,
                 dense_width=512,
                 dropout=0.,  # 0.3,
                 shared=False,
                 max_epochs=500,
                 img_size=(128, 128),
                 #img_size=(256, 256),
                 l2=0.001,
                 keras_filepath=os.path.join('output', 'models',
                                             'gossip-network.hdf5'),
                 warmstart_filepath=None,
                 img_preprocess=None,
                 mask_preprocess=None,
                 from_zero=False,
                 base_models=[],
                 generator_batch_size=8,
                 augment={},
                 max_grad_iterations=1000,
                 ):

        self.img_size = img_size

        # Convolutional layers
        self.conv_num = conv_num
        self.conv_filter_num = conv_filter_num
        self.conv_filter_size = (conv_filter_size, conv_filter_size)
        self.conv_activation = conv_activation
        self.conv_padding = conv_padding
        self.conv_consecutive = conv_consecutive
        self.reciprocal_stimuli = reciprocal_stimuli

        # Max-Pooling layers
        self.pool_size = pool_size
        self.pool_strides = pool_strides

        # Dense layers
        self.dense_stream_num = dense_stream_num
        self.dense_stream_width = dense_stream_width
        self.dense_num = dense_num
        self.dense_width = dense_width
        self.dropout = dropout

        self.shared = shared
        
        # Loss function
        self.l2 = l2

        self.max_epochs = max_epochs
        self.keras_filepath = keras_filepath
        self.warmstart_filepath = warmstart_filepath
        self.img_preprocess = img_preprocess
        self.mask_preprocess = mask_preprocess

        self.generator_batch_size = generator_batch_size

        self.augment = augment
        self.max_grad_iterations = max_grad_iterations

        self.from_zero = from_zero
        self.base_models = base_models
        self.initialize_models()

        self.train = True

    def fit_from_dir(self, path):
        self.opt_network, self.network = self.build_network()
        self.opt_network._make_predict_function()
        self.network._make_predict_function()

        #self.network.summary()

        #self.train = True
        #self.train = False

        if self.train:
            if self.warmstart_filepath is not None:
                try:
                    self.opt_network.load_weights(self.warmstart_filepath)
                    self.network.load_weights(self.warmstart_filepath)
                except:
                    pass

            train_gen = self.get_generator(path, 'train')
            val_gen = self.get_generator(path, 'validation')

            checkpoint = ModelCheckpoint(self.keras_filepath,
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='auto', period=1)
            early_stop = EarlyStopping(patience=50)

            #if os.path.exists(self.keras_filepath):
                #os.remove(self.keras_filepath)
            #elif not os.path.exists(os.path.dirname(self.keras_filepath)):
                #os.makedirs(os.path.dirname(self.keras_filepath))

            tr_steps = len(list(os.walk(os.path.join(path, 'train', 'imgs',
                                                    'seg')))[0][2]) / \
                self.generator_batch_size + 1
            val_steps = len(list(os.walk(os.path.join(path, 'validation',
                                                      'imgs',
                                                      'seg')))[0][2]) / \
                self.generator_batch_size + 1

            #repetitions = 10
            repetitions = 1

            tr_steps *= repetitions
            val_steps *= repetitions

            self.opt_network.fit_generator(train_gen,
                                           steps_per_epoch=tr_steps,
                                           epochs=self.max_epochs,
                                           verbose=2,
                                           callbacks=[early_stop, checkpoint],
                                           validation_data=val_gen,
                                           validation_steps=val_steps,
                                           )

        """
        self.history = self.opt_network.history.history

        self.opt_network.load_weights(self.keras_filepath)
        self.network.load_weights(self.keras_filepath)
        """

        self.opt_network.load_weights(self.keras_filepath)
        self.network.load_weights(self.keras_filepath)

        return self

        IMG_SHAPE = (128, 128)
        
        """
        correct = 0
        incorrect = 0
        for i in range(160, 200):
            best_j = None
            best_out = -np.inf
            for j in range(160, 200):
                img_filename = 'data/partitions/PH2/test/imgs/seg/%04d.bmp' % i
                mask_filename = 'data/partitions/PH2/test/masks/seg/%04d.bmp' % j
        
                img = cv2.resize(cv2.imread(img_filename), IMG_SHAPE)[:, :, ::-1].astype(np.float32) / 255.
                mask = (cv2.resize(cv2.imread(mask_filename, 0), IMG_SHAPE)[:, :, np.newaxis] > 50).astype(np.float32)
        
                pred = self.opt_network.predict({'image': img[np.newaxis],
                                                 'mask': mask[np.newaxis]})
                if pred[0] > best_out:
                    best_out = pred[0]
                    best_j = j
            
            correct += i == best_j
            incorrect += i != best_j
        print(correct, incorrect)
        #return
        """

        return self

    def build_network(self, X=None, y=None):
        return None, None

    def load_weights(self):
        self.opt_network, self.network = self.build_network()

        try:
            self.opt_network.load_weights(self.keras_filepath)
            self.opt_network._make_predict_function()

            self.network.load_weights(self.keras_filepath)
            self.network._make_predict_function()
        except:
            print('Error loading networks')

        return self

    def get_generator(self, path, subset):
        return self.get_base_generator(path, subset)

    def initialize_models(self):
        gen_models = []

        for i, m in enumerate(self.base_models):
            scope_name = 'base-%d' % i
            with tf.variable_scope(scope_name):
                f = open(m, 'rb')
                next_model = pickle.load(f)
                f.close()

                next_model.load_weights()
                gen_models.append((scope_name, next_model))
        self.base_models = gen_models

    def get_base_generator(self, path, subset):
        params = self.augment

        """
        params['preprocessing_function'] = self.img_preprocess
        params = {k: v for k, v in params.items() if k != 'contrast_stretching_perc'}

        #img_gen = MyImageDataGenerator(**params)
        #img_gen = ImageDataGenerator(**params)

        val_params = copy.deepcopy(params)
        val_params['preprocessing_function'] = self.mask_preprocess
        #val_params['contrast_stretching_perc'] = 0

        #mask_gen = MyImageDataGenerator(**val_params)
        mask_gen = ImageDataGenerator(**val_params)
        """

        params['preprocessing_function'] = self.img_preprocess
        img_gen = MyImageDataGenerator(**params)

        val_params = copy.deepcopy(params)
        val_params['preprocessing_function'] = self.mask_preprocess
        val_params['contrast_stretching_perc'] = 0
        mask_gen = MyImageDataGenerator(**val_params)

        seed = 42

        img_gen_ = img_gen.flow_from_directory(
            os.path.join(path, subset, 'imgs'),
            target_size=self.img_size, class_mode=None,
            batch_size=self.generator_batch_size,
            seed=seed, shuffle=False,
            )

        out_gen_ = mask_gen.flow_from_directory(
            os.path.join(path, subset, 'masks'),
            target_size=self.img_size, class_mode=None,
            color_mode='grayscale', batch_size=self.generator_batch_size,
            seed=seed, shuffle=False,
            )

        return izip(img_gen_, out_gen_)

    def transform(self, imgs):
        return imgs
