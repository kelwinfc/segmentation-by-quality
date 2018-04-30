from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt
from models.transformations.transformer import Transformer
import tensorflow as tf
import numpy as np
import cv2


DICE_SMOOTH = 1e-3


def out_dice_coef(y_true, y_pred):
    #return 1. - np.asarray([np.mean(np.abs(gt.ravel() - p.ravel()))
                           #for gt, p in zip(y_true, y_pred)])

    return np.asarray([2. * (np.sum(gt.ravel() * p.ravel()) + DICE_SMOOTH) /
                       (np.sum(gt.ravel()) + np.sum(p.ravel()) + DICE_SMOOTH)
                       for gt, p in zip(y_true, y_pred)])


transfs = [
    'elastic_transform',
    #'flip_transform',
    'morphological_transform',
    'switch_transform',
    'rotate_transform',
    #'offset_transform',
    'flip_offset_transform',
]


def deformed_mask_generator(generator, dataset, seed=42):
    all_dist = []
    all_outputs = []

    mytransfs = [Transformer(dataset, t) for t in transfs]

    random_state = np.random.RandomState(seed)
    
    for batch_id, batch in enumerate(generator):
        img_batch, mask_batch = batch

        img_batch /= 255.
        mask_batch /= 255.

        deformed_mask_batch = mask_batch.copy()

        for i, mask in enumerate(deformed_mask_batch):
            quality = np.random.random()
            transf = random_state.choice(mytransfs)
            deformed_mask_batch[i] = \
                transf.transform(mask[:, :, 0], quality,
                                 random_state)[:, :, np.newaxis]

        inputs_ = {'image': img_batch, 'mask': deformed_mask_batch}

        outputs_ = out_dice_coef(mask_batch, deformed_mask_batch)
        outputs_ = {'output': outputs_}

        #drawing = np.hstack((img_batch[0][:, :, ::-1],
        #                     cv2.merge(3 * [np.hstack((mask_batch[0], deformed_mask_batch[0]))])))
        #cv2.imwrite('metric-mask.png', (255 * drawing).astype(np.uint8))

        yield inputs_, outputs_


def model_based_mask_generator(models, generator, seed=42):
    random_state = np.random.RandomState(seed)

    for batch_id, batch in enumerate(generator):
        img_batch, mask_batch = batch

        model_mask = mask_batch.copy()

        model_idx = np.arange(len(img_batch)) % len(models)
        random_state.shuffle(model_idx)

        for mid in np.arange(len(models)):
            sub_model_idx = model_idx == mid

            if np.any(sub_model_idx):
                with tf.variable_scope(models[mid][0]):
                    preds = models[mid][1].transform(img_batch[sub_model_idx])
                    model_mask[sub_model_idx] = preds

        img_batch /= 255.
        mask_batch /= 255.

        outputs_ = out_dice_coef(mask_batch, model_mask)

        inputs_ = {'image': img_batch, 'mask': model_mask}
        outputs_ = {'output': outputs_}

        yield inputs_, outputs_


def combined_generators(generators, seed=42):
    random_state = np.random.RandomState(seed)

    it_ = 0
    while True:
        next_ = [next(next_gen) for next_gen in generators]
        in_ = [n[0] for n in next_]
        out_ = [n[1] for n in next_]
        
        ret_in = {k: [] for k in in_[0].keys()}
        ret_out = {k: [] for k in out_[0].keys()}
        
        for i in range(len(generators)):
            for k in ret_in.keys():
                ret_in[k] += list(in_[i][k])
            for k in ret_out.keys():
                ret_out[k] += list(out_[i][k])

        for k in ret_in.keys():
            ret_in[k] = np.asarray(ret_in[k])
        for k in ret_out.keys():
            ret_out[k] = np.asarray(ret_out[k])

        yield ret_in, ret_out

        it_ += 1
        it_ = it_ % len(generators)


def deformed_mask2mask_generator(generator, dataset, seed=42):
    all_dist = []
    all_outputs = []

    mytransfs = [Transformer(dataset, t) for t in transfs]

    random_state = np.random.RandomState(seed)
    
    for batch_id, batch in enumerate(generator):
        img_batch, mask_batch = batch
        deformed_mask_batch = np.zeros_like(mask_batch)

        img_batch /= 255.
        mask_batch /= 255.

        for i, mask in enumerate(mask_batch):
            quality = np.random.random()
            transf = random_state.choice(mytransfs)
            deformed_mask_batch[i] = transf.transform(mask[:, :, 0], quality, random_state)[:, :, np.newaxis]

        inputs_ = {'image': img_batch, 'mask': deformed_mask_batch}

        outputs_ = {'output': mask_batch - deformed_mask_batch,
                    'corrected-output': mask_batch}

        yield inputs_, outputs_
