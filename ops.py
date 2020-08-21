import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

# try:
#     image_summary = tf.image_summary
#     scalar_summary = tf.scalar_summary
#     histogram_summary = tf.histogram_summary
#     merge_summary = tf.merge_summary
#     SummaryWriter = tf.train.SummaryWriter
# except:
image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

# class batch_norm(object):
#     def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
#         with tf.variable_scope(name):
#             self.epsilon  = epsilon
#             self.momentum = momentum
#             self.name = name
#
#     def __call__(self, x, train=True):
#         return tf.contrib.layers.batch_norm(x,
#                                             decay=self.momentum,
#                                             updates_collections=None,
#                                             epsilon=self.epsilon,
#                                             scale=True,
#                                             is_training=train,
#                                             scope=self.name)
def batch_norm(x,epsilon=1e-4, momentum = 0.99, name="batch_norm",train=True):
    '''
    可以选择比较大的初始学习率，让你的训练速度飙涨。
    移除或使用较低的drop out、L2正则项参数。
    b 可以忽略了，因为b 的作用其实被 β 代替了
    '''
    return tf.contrib.layers.batch_norm(x,
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=epsilon,
                                        scale=True,
                                        is_training=train,
                                        scope=name)


def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                                        decay=self.momentum,
                                        updates_collections=None,
                                        epsilon=self.epsilon,
                                        scale=True,
                                        is_training=train,
                                        scope=self.name)


def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_,
           output_dim, # 内核深度
           k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, # 默认内核尺寸:3x3 移动步长2x2
           name="conv2d"):
    '对input_进行卷积，'
    with tf.variable_scope(name):
        # 定义卷积核
        w = tf.get_variable('w', [k_h, k_w, #内核尺寸
                                  input_.get_shape()[-1],# 当前层节点矩阵深度
                                  output_dim], # 卷积后的深度
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
                            # initializer = tf.constant_initializer(0.125))
        conv = tf.nn.conv2d(input_, #当前层节点矩阵
                            w, #卷积核[filter_height, filter_width, in_channels, out_channels]
                            strides=[1, d_h, d_w, 1], #不同维度上的步长(第一维和第四维必须是1,因为卷积层的步长只对长和宽有用)
                            padding='SAME')
        #定义偏置项,偏置项个数与内核的深度个数相同
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape()) #bias_add用于给每个节点加上偏置项 #(10,40,64)
        conv = tf.nn.bias_add(conv, biases)
        return conv

def resblock(input_, k_h=3, k_w=3, d_h=1, d_w=1, name = "resblock"):
    '将input_两次卷积并batch_norm后与自己相加'
    conv1 = lrelu(batch_norm(conv2d(input_, input_.get_shape()[-1],
                                    k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w,
                                    name=name+"_conv1"),name=name+"_bn1"))
    conv2 = lrelu(batch_norm(conv2d(conv1, input_.get_shape()[-1],
                            k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w,
                            name=name+"_conv2"),name=name+"_bn2"))
    # return conv2
    return tf.add(input_, conv2) #对应位置相加

def deconv2d(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    '''
    对input_进行一次反卷积后输出
    '''
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, #内核尺寸默认:3x3
                                  output_shape[-1],# 反卷积过后的深度
                                  input_.get_shape()[-1]], # 当前层节点矩阵深度
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, #卷积核，[filter_height, filter_width, out_channels卷积核个数, in_channels图像通道数]
                                            output_shape=output_shape, #反卷积操作输出的shape,卷积操作没有这个参数,这里是为了考虑到反卷积后出现奇/偶维度问题
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def gradientweight():
    kernels = []
    # [np.array([
    # [-1,0,1],
    # [-1,0,1],
    # [-1,0,1]])]
    # kernels.append(np.array([
    #     [-1,-1,-1],
    #     [0,0,0],
    #     [1,1,1]]))
    # kernels.append(np.array([
    #     [0,-1,0],
    #     [-1,4,-1],
    #     [0,-1,0]]))
    kernels.append(np.array([
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]]))
    weights = []
    for i in range(0,3): #input channel
        weightPerChannel = []
        for j in range(0,1):#kernel
            weightPerKernel = []
            for k in range(0,i):#before zero
                weightPerKernel.append(np.zeros([3,3]))
            weightPerKernel.append(kernels[j])
            for k in range(i,3-1):#after zero
                weightPerKernel.append(np.zeros([3,3]))
            weightPerChannel.extend(weightPerKernel)
        weights.append(weightPerChannel)
    weights = np.array(weights).transpose(2,3,0,1)
    return weights

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x, name=name)

def Dropout(x, keep_prob=0.5, is_training=True):
    '''
    使用dropout避免过拟合
    :param x:
    :param keep_prob: 每个元素被保留下来的概率
    :param is_training: if None, will use the current context by default.
    :return:
    '''
    keep_prob = tf.constant(keep_prob if is_training else 1.0)
    return tf.nn.dropout(x, keep_prob)
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    '''
    神经网络中的线性化运算:将ipnput_与一个随机数矩阵相乘输出
    '''
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
                                 # tf.ones_initializer())
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
def symL1(img):
    '''
    计算脸的对称损失
    '''
    width = img.get_shape().as_list()[2]
    right = img[:,:,int(width/2):,:]
    left = img[:,:,0:int(width/2),:]
    # left = tf.stop_gradient(img[:,:,0:int(width/2),:])
    # right1 = tf.concat((right[:,:,::-1,:],right), axis=2)
    # left1 = tf.concat((left,left[:,:,::-1,:]), axis=2)
    # return tf.maximum(right1, left1)
    return tf.abs(left - right[:,:,::-1,:])

def total_variation(images, name=None):
    '''
    计算图像的全变分，用于图像降噪。通常有噪声图像的全变分比无噪声图像的全变分明显大。
    通过最小化全变峰可以消除噪声。
    '''
    """Calculate and return the Total Variation for one or more images.

    The total variation is the sum of the absolute differences for neighboring
    pixel-values in the input images. This measures how much noise is in the images.

    This can be used as a loss-function during optimization so as to suppress noise
    in images. If you have a batch of images, then you should calculate the scalar
    loss-value as the sum: `loss = tf.reduce_sum(tf.image.total_variation(images))`

    This implements the anisotropic 2-D version of the formula described here:

    https://en.wikipedia.org/wiki/Total_variation_denoising

    Args:
        images: 4-D Tensor of shape `[batch, height, width, channels]` or
                3-D Tensor of shape `[height, width, channels]`.

        name: A name for the operation (optional).

    Raises:
        ValueError: if images.shape is not a 3-D or 4-D vector.

    Returns:
        The total variation of `images`.

        If `images` was 4-D, a 1-D float Tensor of shape `[batch]` with the
        total variation for each image in the batch.
        If `images` was 3-D, a scalar float with the total variation for that image.
    """

    with ops.name_scope(name, 'total_variation'):
        ndims = images.get_shape().ndims

        if ndims == 3:
            # The input is a single image with shape [height, width, channels].

            # Calculate the difference of neighboring pixel-values.
            # The images are shifted one pixel along the height and width by slicing.
            pixel_dif1 = images[1:,:,:] - images[:-1,:,:]
            pixel_dif2 = images[:,1:,:] - images[:,:-1,:]

            # Sum for all axis. (None is an alias for all axis.)
            sum_axis = None
        elif ndims == 4:
            # The input is a batch of images with shape [batch, height, width, channels].

            # Calculate the difference of neighboring pixel-values.
            # The images are shifted one pixel along the height and width by slicing.
            pixel_dif1 = images[:,1:,:,:] - images[:,:-1,:,:]
            pixel_dif2 = images[:,:,1:,:] - images[:,:,:-1,:]

            # Only sum for the last 3 axis.
            # This results in a 1-D tensor with the total variation for each image.
            sum_axis = [1, 2, 3]
        else:
            raise ValueError('\'images\' must be either 3 or 4-dimensional.')

        # Calculate the total variation by taking the absolute value of the
        # pixel-differences and summing over the appropriate axis.
        tot_var = tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) + \
                  tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis)

    return tot_var