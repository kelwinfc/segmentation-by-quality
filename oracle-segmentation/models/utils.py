from keras.losses import binary_crossentropy
from keras import backend as K

DICE_SMOOTH = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + DICE_SMOOTH) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + DICE_SMOOTH)
        #(K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + DICE_SMOOTH)


def dice_coef_sqr(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.square(K.sum(y_true_f * K.square(y_pred_f)))
    return (2. * intersection + DICE_SMOOTH) / \
        (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + DICE_SMOOTH)


def dice_coef_loss(y_true, y_pred):
    #return -dice_coef(y_true, y_pred) -dice_coef(1 - y_true, 1 - y_pred)
    return 1. - dice_coef(y_true, y_pred)


def dice_coef_loss_plus_mse(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    return dice_coef_loss(y_true, y_pred) + K.mean(K.square(y_true_f -
                                                            y_pred_f))

def regularized_dice_loss(ytrue, ypred):
    y_true_f = 0.05 + 0.95 * K.flatten(ytrue)
    y_pred_f = K.flatten(ypred)
    intersection = K.sum(y_true_f * y_pred_f)
    ret = (2. * intersection + DICE_SMOOTH) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + DICE_SMOOTH)
    return 1 - ret

def weighted_dice_loss(pos_rate):
    def wdl(ytrue, ypred):
        pos = dice_coef(ytrue, ypred)
        neg = dice_coef(1. - ytrue, 1. - ypred)
        return 1. - (pos_rate * neg + (1. - pos_rate) * pos)
    return wdl


def macro_dice_coef(ytrue, ypred):
    ret = 0.
    nclasses = ypred.get_shape()[-1].value

    for class_ in range(nclasses):
        next_ret = dice_coef(ytrue[:, :, :, class_], ypred[:, :, :, class_])
        ret += next_ret

    return ret / float(nclasses)


def macro_balanced_dice_coef(ytrue, ypred):
    ret = 0.
    nclasses = ypred.get_shape()[-1].value

    for class_ in range(nclasses):
        next_ret = dice_coef(ytrue[:, :, :, class_], ypred[:, :, :, class_])
        next_ret += dice_coef(1. - ytrue[:, :, :, class_],
                              1. - ypred[:, :, :, class_])
        ret += 0.5 * next_ret

    return ret / float(nclasses)


def macro_balanced_dice_coef_loss(ytrue, ypred):
    return 1. - macro_balanced_dice_coef(ytrue, ypred)


def macro_dice_coef_loss(ytrue, ypred):
    return 1. - macro_dice_coef(ytrue, ypred)


def worst_dice_coef(ytrue, ypred):
    ret = dice_coef(ytrue[:, :, :, 0], ypred[:, :, :, 0])
    nclasses = ypred.get_shape()[-1].value

    for class_ in range(1, nclasses):
        next_ret = dice_coef(ytrue[:, :, :, class_], ypred[:, :, :, class_])
        ret = K.minimum(ret, next_ret)

    return ret


def worst_balanced_dice_coef(ytrue, ypred):
    ret = dice_coef(ytrue[:, :, :, 0], ypred[:, :, :, 0])
    nclasses = ypred.get_shape()[-1].value

    for class_ in range(1, nclasses):
        next_ret = dice_coef(ytrue[:, :, :, class_], ypred[:, :, :, class_])
        next_ret += dice_coef(1. - ytrue[:, :, :, class_],
                              1. - ypred[:, :, :, class_])
        ret = K.minimum(ret, 0.5 * next_ret)

    return ret


def worst_balanced_dice_coef_loss(ytrue, ypred):
    return 1. - worst_balanced_dice_coef(ytrue, ypred)

                                
def worst_dice_coef_loss(ytrue, ypred):
    return 1. - worst_dice_coef(ytrue, ypred)


def worst_times_best(ytrue, ypred):
    return 1. - worst_dice_coef(ytrue, ypred) * best_dice_coef(ytrue, ypred)


def best_dice_coef(ytrue, ypred):
    ret = dice_coef(ytrue[:, :, :, 0], ypred[:, :, :, 0])
    nclasses = ypred.get_shape()[-1].value

    for class_ in range(1, nclasses):
        next_ret = dice_coef(ytrue[:, :, :, class_], ypred[:, :, :, class_])
        ret = K.maximum(ret, next_ret)

    return ret

def dice_coef_class(ytrue, ypred, class_):
    return dice_coef(ytrue[:, :, :, class_],
                     ypred[:, :, :, class_])


def geometric_dice(ytrue, ypred):
    ret = 1.
    nclasses = ypred.get_shape()[-1].value

    for class_ in range(nclasses):
        next_ret = dice_coef(ytrue[:, :, :, class_], ypred[:, :, :, class_])
        ret *= (0.5 + 0.5 * next_ret)

    return ret


def geometric_dice_loss(ytrue, ypred):
    return 1. - geometric_dice(ytrue, ypred)

def average_crossentropy(ytrue, ypred):
    ret = 0.
    nclasses = ypred.get_shape()[-1].value

    for class_ in range(nclasses):
        next_ret = binary_crossentropy(ytrue[:, :, :, class_],
                                       ypred[:, :, :, class_])
        ret += next_ret

    return ret
