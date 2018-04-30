import numpy as np
import pickle
import cv2
import os

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
from keras.backend.tensorflow_backend import set_session

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
set_session(sess)

from models.metric_learning import MetricBasedSegmentationNetwork, \
    SingleStreamMetricSegmentationNetwork, DualStreamMetricSegmentationNetwork
from models.gossip_unet import GossipUNet


def preprocess(img):
    return cv2.resize(img, (128, 128))


def dice(m1, m2):
    m1 = m1.flatten() / 255.
    m2 = m2.flatten()

    m1 = m1.round()
    m2 = m2.round()

    return 2 * (np.sum(m1 * m2) + 1.) / (np.sum(m1) + np.sum(m2) + 1.)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_augmentation = {'horizontal_flip': True,
                     #'vertical_flip': True,
                     'zoom_range': 0.2,
                     'width_shift_range': 0.1,
                     'height_shift_range': 0.1,
                     'shear_range': 0.1,
                     'fill_mode': 'reflect'}

models = [#os.path.join('output', 'models', 'unet-2.pckl'),
          #os.path.join('output', 'models', 'unet-3.pckl'),
          #os.path.join('output', 'models', 'unet-4.pckl'),
          os.path.join('output', 'models', 'unet-2-0.pckl'),
          os.path.join('output', 'models', 'unet-2-1.pckl'),
          os.path.join('output', 'models', 'unet-2-2.pckl'),
          os.path.join('output', 'models', 'unet-2-3.pckl'),
          os.path.join('output', 'models', 'unet-2-4.pckl'),
          ]

path = os.sys.argv[1]

with tf.variable_scope('metric'):
    #kpath = os.path.join('output', 'models', 'keras', 'gossip-unet.hdf5')
    #model = GossipUNet(augment=data_augmentation,
                       #keras_filepath=kpath)

    kpath = os.path.join('output', 'models', 'keras', 'gossipnet.hdf5')
    model = MetricBasedSegmentationNetwork(augment=data_augmentation,
                                           keras_filepath=kpath)
    #model = MetricBasedSegmentationNetwork(base_models=models,
                                          #augment=data_augmentation,
                                          #keras_filepath=kpath)
    #model = MetricBasedSegmentationNetwork(models,
                                            #augment=data_augmentation,
                                            #keras_filepath=kpath,
                                            #max_grad_iterations=0)

    #kpath = os.path.join('output', 'models', 'keras', 'singlenet.hdf5')
    #model = SingleStreamMetricSegmentationNetwork(augment=data_augmentation,
                                                  #keras_filepath=kpath)

    #kpath = os.path.join('output', 'models', 'keras', 'dualnet.hdf5')
    #model = DualStreamMetricSegmentationNetwork(augment=data_augmentation,
                                                #keras_filepath=kpath)

    if True:
        model.fit_from_dir(path)#'data/partitions/PH2')

ipath = os.path.join(path, 'test', 'imgs', 'seg')
mpath = os.path.join(path, 'test', 'masks', 'seg')

imgs = sorted(list(os.walk(ipath))[0][2])
masks = sorted(list(os.walk(mpath))[0][2])

imgs = [preprocess(cv2.imread(os.path.join(ipath, i))[:, :, ::-1]) for i in imgs]
masks = [preprocess(cv2.imread(os.path.join(mpath, i), 0))[:, :, np.newaxis]
         for i in masks]

"""
for m in models:
    print(m)
    
    f = open(m, 'rb')
    next_model = pickle.load(f)
    f.close()

    next_model.load_weights()

    preds = next_model.transform(np.asarray(imgs))
    pdice = [dice(m, p) for m, p in zip(masks, preds)]
    
    print(pdice)
    print(np.mean(pdice))
    print()
"""

print(len(imgs))

preds = model.transform(imgs[:20])

for ix, (i, p) in enumerate(zip(imgs, preds)):
    i2 = i.copy().astype(np.float)
    i2 *= p[:, :, np.newaxis]
    cv2.imwrite('pred-%d.png' % ix,
                np.hstack((i, i2)).astype(np.uint8)[:, :, ::-1])

pdice = [dice(m, p) for m, p in zip(masks[:20], preds)]
print(sorted(pdice))
print(np.mean(pdice))

"""
f = open(os.path.join('output', 'models', 'unet.pckl'), 'wb')
model.network = None
model.opt_network = None
pickle.dump(model, f)
f.close()
"""
