import tensorflow as tf

# weight variable
def weightVariable(shape, name):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    weight = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return weight

# bias variable
def biasVariable(shape, name):
    initializer = tf.constant_initializer(0.1)
    bias = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return bias

# 2d convolutional layer
def conv2d(x, W, strides = [1,2,2,1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

# deconv layer
def deConv3d(y, w, outShape, strides=[1,2,2,2,1]):
    return tf.nn.conv3d_transpose(y, w, output_shape = outShape, strides=strides, padding='SAME')