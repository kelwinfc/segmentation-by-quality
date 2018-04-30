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
base_models_path = os.path.join('output', 'models', dataset)
gossip_model = os.sys.argv[2]
base_models = os.sys.argv[3:]
base_models = [os.path.join(base_models_path, x) for x in base_models]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_augmentation = {'horizontal_flip': True,
                     #'vertical_flip': True,
                     'zoom_range': 0.2,
                     'width_shift_range': 0.1,
                     'height_shift_range': 0.1,
                     'shear_range': 0.1,
                     'fill_mode': 'reflect'}

val_ipath = os.path.join('data', 'partitions', dataset, 'validation', 'imgs', 'seg')
val_mpath = os.path.join('data', 'partitions', dataset, 'validation', 'masks', 'seg')

ipath = os.path.join('data', 'partitions', dataset, 'test', 'imgs', 'seg')
mpath = os.path.join('data', 'partitions', dataset, 'test', 'masks', 'seg')

val_imgs = sorted(list(os.walk(val_ipath))[0][2])
val_masks = sorted(list(os.walk(val_mpath))[0][2])

imgs = sorted(list(os.walk(ipath))[0][2])
masks = sorted(list(os.walk(mpath))[0][2])

val_imgs = [preprocess(cv2.imread(os.path.join(val_ipath, i))[:, :, ::-1])
            for i in val_imgs]
val_masks = [preprocess(cv2.imread(os.path.join(val_mpath, i), 0))[:, :, np.newaxis]
             for i in val_masks]

imgs = [preprocess(cv2.imread(os.path.join(ipath, i))[:, :, ::-1]) for i in imgs]
masks = [preprocess(cv2.imread(os.path.join(mpath, i), 0))[:, :, np.newaxis]
         for i in masks]

d_num = 1
d_width = 512
conv_num = 4
base_keras_path = os.path.join('output', 'models', dataset, 'keras')
kpath = os.path.join(base_keras_path, '%s-%d-%d-%d.hdf5' % ('linear-gossip',
                                                            conv_num,
                                                            d_num, d_width))
print(kpath + '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

pckl_path = os.path.join('output', 'models', dataset,
                        '%s-%d-%d-%d.pckl' %
                        ('linear-gossip', conv_num, d_num, d_width))
gossip_model = MetricBasedSegmentationNetwork(
                                              base_models=base_models,
                                              conv_num=conv_num,
                                              reciprocal_stimuli=0.5,
                                              dense_stream_num=d_num,
                                              dense_stream_width=d_width,
                                              dense_num=d_num,
                                              dense_width=d_width,
                                              max_epochs=500,
                                              max_grad_iterations=0,
                                              l2=0.001,
                                              keras_filepath=kpath)

gossip_model.train = False
gossip_model.fit_from_dir(os.path.join('data', 'partitions', dataset))

model_preds = {}

pred_qualities = -np.inf * np.ones(len(imgs))
real_qualities = -np.inf * np.ones(len(imgs))
final_masks = [None for _ in imgs]
avg_masks = [np.zeros((128, 128)) for _ in imgs]
gossip_masks = [np.zeros((128, 128)) for _ in imgs]

w_avg_masks = np.zeros(len(imgs))
w_gossip_masks = np.zeros(len(imgs))

for name in base_models:
    print(name)

    f = open(name, 'rb')
    next_model = pickle.load(f)
    f.close()

    next_model.load_weights()

    val_preds = next_model.transform(np.asarray(val_imgs))
    val_q = gossip_model.predict_quality(val_imgs, [255 * m.round() for m in val_preds])
    val_dice = np.asarray([dice(m, p) for m, p in zip(val_masks, val_preds)])

    print('Validation:', val_q.mean(), val_dice.mean())

    preds = next_model.transform(np.asarray(imgs))
    model_preds[name] = preds

    from scipy.stats import kendalltau
    alpha = (1. + kendalltau(val_dice, val_q)[0]) / 2.

    next_q = gossip_model.predict_quality(imgs, [255 * m.round() for m in preds])
    next_q = (next_q - np.mean(val_q)) / np.std(val_q) * np.std(val_dice) + np.mean(val_dice)
    next_q = (1 - alpha) * val_dice + alpha * next_q

    #next_q = np.minimum(1., np.maximum(0.8, next_q))
    pdice = np.asarray([dice(m, p) for m, p in zip(masks, preds)])
    print(pdice)
    print(next_q)
    print('Validation', np.mean(val_dice), np.std(val_dice))
    print('Predicted', np.mean(next_q), np.std(next_q))
    print('Test', np.mean(pdice), np.std(pdice))

    for i, (m, predq, prevq) in enumerate(zip(preds, next_q, pred_qualities)):
        avg_masks[i] += m[:, :, 0] * np.mean(val_dice)
        gossip_masks[i] += m[:, :, 0] * predq

        w_avg_masks[i] += np.mean(val_dice)
        w_gossip_masks[i] += predq

        if predq > prevq:
            pred_qualities[i] = predq
            final_masks[i] = m

            real_qualities[i] = pdice[i]

    print(name, np.mean(pdice))

avg_masks = np.asarray(avg_masks) / w_avg_masks[:, np.newaxis, np.newaxis]
gossip_masks = np.asarray(gossip_masks) / w_gossip_masks[:, np.newaxis, np.newaxis]

print()
print()
print()

pdice = [dice(m, p) for m, p in zip(masks, final_masks)]
print('Gossip Ensemble', np.mean(pdice))

pdice = [dice(m, p) for m, p in zip(masks, gossip_masks)]
print('W-Gossip Ensemble', np.mean(pdice))

pdice = [dice(m, p) for m, p in zip(masks, avg_masks)]
print('Mean Ensemble', np.mean(pdice))
