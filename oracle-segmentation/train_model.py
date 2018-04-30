import pickle
import cv2
import os
import sys
import json

if len(sys.argv) < 3:
    print("Usage: %s MODEL DATASET" % sys.argv[0])
    print("(MODEL can be UNet, for example)")
    sys.exit(-1)

import numpy as np
np.random.seed(1337)

import tensorflow as tf
tf.set_random_seed(2)

import random as rn
rn.seed(12345)

#single thread
session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True
session_conf.inter_op_parallelism_threads = 1
session_conf.intra_op_parallelism_threads = 1

from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

MODEL = sys.argv[1]
DATASET = sys.argv[2]

import importlib
mod = importlib.import_module('models.' + MODEL.lower())
Model = getattr(mod, MODEL)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('data/' + DATASET + '.json') as f:
    conf = json.load(f)

data_augmentation = conf['augment']
data_augmentation['rescale'] = 1. / 255.

for num_conv in [2, 3, 4]:
    base_path = os.path.join('output', 'models', DATASET)
    base_keras_path = os.path.join(base_path, 'keras')

    if not os.path.exists(base_keras_path):
        os.makedirs(base_keras_path)

    pckl_path = os.path.join(base_path, '%s-%d.pckl' % (MODEL.lower(),
                                                        num_conv))
    if os.path.exists(pckl_path):
        print('Model %s exists' % pckl_path)
        continue

    model = Model(conv_num=num_conv,
                  augment=data_augmentation,
                  keras_filepath=os.path.join(base_keras_path,
                                              '%s-%d.hdf5' % (MODEL.lower(),
                                                              num_conv)))

    model.fit_from_dir('data/partitions/' + DATASET)

    f = open(pckl_path, 'wb')
    model.network = None
    model.opt_network = None
    pickle.dump(model, f)
    f.close()
