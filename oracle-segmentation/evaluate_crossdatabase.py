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


src_dataset = os.sys.argv[1]
dst_dataset = os.sys.argv[2]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_augmentation = {'horizontal_flip': True,
                     #'vertical_flip': True,
                     'zoom_range': 0.2,
                     'width_shift_range': 0.1,
                     'height_shift_range': 0.1,
                     'shear_range': 0.1,
                     'fill_mode': 'reflect'}

base_models_path = os.path.join('output', 'models', src_dataset)
#base_models = [os.path.join(base_models_path, m)
               #for m in list(os.walk(base_models_path))[0][2]]

if len(os.sys.argv) < 4:
    base_models = []
else:
    base_models = [os.sys.argv[3],]

base_models = [os.path.join(base_models_path, x) for x in base_models]

ipath = os.path.join('data', 'partitions', dst_dataset, 'test', 'imgs', 'seg')
mpath = os.path.join('data', 'partitions', dst_dataset, 'test', 'masks', 'seg')

imgs = sorted(list(os.walk(ipath))[0][2])
masks = sorted(list(os.walk(mpath))[0][2])

imgs = [preprocess(cv2.imread(os.path.join(ipath, i))[:, :, ::-1]) for i in imgs]
masks = [preprocess(cv2.imread(os.path.join(mpath, i), 0))[:, :, np.newaxis]
         for i in masks]

for m in base_models:
    print(m)
    
    f = open(m, 'rb')
    next_model = pickle.load(f)
    f.close()

    next_model.load_weights()

    preds = next_model.transform(np.asarray(imgs))
    pdice = [dice(m, p) for m, p in zip(masks, preds)]

    #print(pdice)
    print(np.mean(pdice))
    print()

print(len(imgs))

for d_num in [1, ]:
    for d_width in [512]:  # [256, 512]:
        for conv_num in [4,]:#[3, 4]:
            for single, single_name in [(False, 'linear-gossip'), (True, 'single')]:
                if single:
                    continue

                base_keras_path = os.path.join('output', 'models', src_dataset,
                                            'keras')
                #if not os.path.exists(base_keras_path):
                #    os.makedirs(base_keras_path)

                kpath = os.path.join(base_keras_path,
                                    '%s-%d-%d-%d.hdf5' % (single_name,
                                                          conv_num,
                                                          d_num, d_width))
                print(kpath + '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                pckl_path = os.path.join('output', 'models', src_dataset,
                                        '%s-%d-%d-%d.pckl' %
                                        (single_name, conv_num, d_num, d_width))

                if not os.path.exists(kpath):
                    print('Model %s does not exist' % pckl_path)
                    continue

                model = MetricBasedSegmentationNetwork(
                                                    base_models=base_models,
                                                    conv_num=conv_num,
                                                    reciprocal_stimuli=0.5,
                                                    dense_stream_num=d_num,
                                                    dense_stream_width=d_width,
                                                    dense_num=d_num,
                                                    dense_width=d_width,
                                                    max_epochs=500,
                                                    max_grad_iterations=100,
                                                    l2=0.001,
                                                    keras_filepath=kpath)

                model.train = False
                model.fit_from_dir(os.path.join('data', 'partitions', src_dataset))

                preds = model.transform(imgs)

                pdice = [dice(m, p) for m, p in zip(masks, preds)]
                #print(sorted(pdice))
                print('%.4f' % (100 * np.mean(pdice)))

"""
f = open(os.path.join('output', 'models', 'unet.pckl'), 'wb')
model.network = None
model.opt_network = None
pickle.dump(model, f)
f.close()
"""
