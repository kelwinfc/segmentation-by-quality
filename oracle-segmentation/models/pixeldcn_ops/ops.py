# THis code comes from:
# https://github.com/HongyangGao/PixelDCN

import tensorflow as tf
HACK = 0


def dilate_tensor(inputs, axes, shifts, scope):
    for index, axis in enumerate(axes):
        eles = tf.unstack(inputs, axis=axis, name=scope+'/unstack%s' % index)
        zeros = tf.zeros_like(
            eles[0], dtype=tf.float32, name=scope+'/zeros%s' % index)
        for ele_index in range(len(eles), 0, -1):
            eles.insert(ele_index-shifts[index], zeros)
        inputs = tf.stack(eles, axis=axis, name=scope+'/stack%s' % index)
    return inputs


def conv2d(inputs, out_num, kernel_size, scope, stride=1, d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d(
        inputs, out_num, kernel_size, scope=scope, stride=stride,
        data_format=d_format, activation_fn=None, biases_initializer=None)
    return outputs


def ipixel_dcl(inputs, out_num, kernel_size, scope, activation_fn=tf.nn.relu,
               d_format='NHWC'):
    global HACK
    scope += str(HACK)
    HACK += 1
    """
    inputs: input tensor
    out_num: output channel number
    kernel_size: convolutional kernel size
    scope: operation scope
    activation_fn: activation function, could be None if needed
    """
    axis = (d_format.index('H'), d_format.index('W'))
    channel_axis = d_format.index('C')
    conv1 = conv2d(inputs, out_num, kernel_size,
                   scope+'/conv1', d_format=d_format)
    conv1_concat = tf.concat(
        [inputs, conv1], channel_axis, name=scope+'/concat1')
    conv2 = conv2d(conv1_concat, out_num, kernel_size,
                   scope+'/conv2', d_format=d_format)
    conv2_concat = tf.concat(
        [conv1_concat, conv2], channel_axis, name=scope+'/concat2')
    conv3 = conv2d(conv2_concat, 2*out_num, kernel_size,
                   scope+'/conv3', d_format=d_format)
    conv4, conv5 = tf.split(conv3, 2, channel_axis, name=scope+'/split')
    dialte1 = dilate_tensor(conv1, axis, (0, 0), scope+'/dialte1')
    dialte2 = dilate_tensor(conv2, axis, (1, 1), scope+'/dialte2')
    dialte3 = dilate_tensor(conv4, axis, (1, 0), scope+'/dialte3')
    dialte4 = dilate_tensor(conv5, axis, (0, 1), scope+'/dialte4')
    outputs = tf.add_n([dialte1, dialte2, dialte3, dialte4], scope+'/add')
    if activation_fn:
        outputs = activation_fn(outputs)
    return outputs

