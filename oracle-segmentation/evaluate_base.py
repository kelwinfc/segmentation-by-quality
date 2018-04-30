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
#base_models = [os.path.join(base_models_path, m)
               #for m in list(os.walk(base_models_path))[0][2]]
base_models = ["dilatednet-2.pckl", "dilatednet-3.pckl", "dilatednet-4.pckl",
               "unet-2.pckl", "unet-3.pckl", "unet-4.pckl"]

base_models = [os.path.join(base_models_path, x) for x in base_models]

val_ipath = os.path.join('data', 'partitions', dataset, 'validation', 'imgs', 'seg')
val_mpath = os.path.join('data', 'partitions', dataset, 'validation', 'masks', 'seg')

ts_ipath = os.path.join('data', 'partitions', dataset, 'test', 'imgs', 'seg')
ts_mpath = os.path.join('data', 'partitions', dataset, 'test', 'masks', 'seg')

val_imgs = sorted(list(os.walk(val_ipath))[0][2])
val_masks = sorted(list(os.walk(val_mpath))[0][2])

ts_imgs = sorted(list(os.walk(ts_ipath))[0][2])
ts_masks = sorted(list(os.walk(ts_mpath))[0][2])

val_imgs = [preprocess(cv2.imread(os.path.join(val_ipath, i))[:, :, ::-1]) for i in val_imgs]
val_masks = [preprocess(cv2.imread(os.path.join(val_mpath, i), 0))[:, :, np.newaxis] for i in val_masks]

ts_imgs = [preprocess(cv2.imread(os.path.join(ts_ipath, i))[:, :, ::-1]) for i in ts_imgs]
ts_masks = [preprocess(cv2.imread(os.path.join(ts_mpath, i), 0))[:, :, np.newaxis] for i in ts_masks]

for name in base_models:
    print(name)
    
    f = open(name, 'rb')
    next_model = pickle.load(f)
    f.close()

    next_model.load_weights()

    val_preds = next_model.transform(np.asarray(val_imgs))
    val_pdice = [dice(m, p) for m, p in zip(val_masks, val_preds)]
    
    ts_preds = next_model.transform(np.asarray(ts_imgs))
    ts_pdice = [dice(m, p) for m, p in zip(ts_masks, ts_preds)]

    #print(pdice)
    print('%s %.4f %.4f' % (name, 100 * np.mean(val_pdice), 100 * np.mean(ts_pdice)))
    print()
