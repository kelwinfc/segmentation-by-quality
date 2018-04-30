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
#from models.gossip_unet import GossipUNet


def preprocess(img):
    return cv2.resize(img, (128, 128))


def dice(m1, m2):
    m1 = m1.flatten() / 255.
    m2 = m2.flatten()

    m1 = m1.round()
    m2 = m2.round()

    return 2 * (np.sum(m1 * m2) + 1.) / (np.sum(m1) + np.sum(m2) + 1.)


dataset = os.sys.argv[1]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_augmentation = {'horizontal_flip': True,
                     #'vertical_flip': True,
                     'zoom_range': 0.2,
                     'width_shift_range': 0.1,
                     'height_shift_range': 0.1,
                     'shear_range': 0.1,
                     'fill_mode': 'reflect'}

base_models_path = os.path.join('output', 'models', dataset)
base_models = [os.path.join(base_models_path, m)
               for m in list(os.walk(base_models_path))[0][2]]

for d_num in [1,]:# 2]:
    for d_width in [512]:  # [256, 512]:
        for conv_num in [4,]:#[3, 4]:
            for single, single_name in [(False, 'linear-gossip'), (True, 'single')]:
                if single:
                    continue

                base_keras_path = os.path.join('output', 'models', dataset,
                                            'keras')
                if not os.path.exists(base_keras_path):
                    os.makedirs(base_keras_path)

                kpath = os.path.join(base_keras_path,
                                    '%s-%d-%d-%d.hdf5' % (single_name,
                                                          conv_num,
                                                          d_num, d_width))

                pckl_path = os.path.join('output', 'models', dataset,
                                        '%s-%d-%d-%d.pckl' %
                                        (single_name, conv_num, d_num, d_width))

                if os.path.exists(pckl_path):
                    print('Model %s exists' % pckl_path)
                    continue

                if single:
                    model = SingleStreamMetricSegmentationNetwork(conv_num=conv_num,
                                                        reciprocal_stimuli=1.,
                                                        dense_stream_num=d_num,
                                                        dense_stream_width=d_width,
                                                        dense_num=d_num,
                                                        dense_width=d_width,
                                                        max_epochs=500,
                                                        max_grad_iterations=500,
                                                        l2=0.001,
                                                        keras_filepath=kpath)
                else:
                    model = MetricBasedSegmentationNetwork(
                                                        conv_num=conv_num,
                                                        reciprocal_stimuli=1.,
                                                        dense_stream_num=d_num,
                                                        dense_stream_width=d_width,
                                                        dense_num=d_num,
                                                        dense_width=d_width,
                                                        max_epochs=500,
                                                        max_grad_iterations=500,
                                                        l2=0.0001,
                                                        keras_filepath=kpath)

                model.fit_from_dir(os.path.join('data', 'partitions', dataset))

                f = open(pckl_path, 'wb')
                model.network = None
                model.opt_network = None
                pickle.dump(model, f)
                f.close()

os.sys.exit(0)
from_zero=False,
base_models=[],
max_grad_iterations=1000,
# Alternatives:
# Train the one-stream model
# Train the gossip network

print(base_models)
os.sys.exit(0)

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
