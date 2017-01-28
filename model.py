from ops import *
import sys


# this method saves the model
def saveModel(sess, savePath):
    saver = tf.train.Saver()
    saver.save(sess, savePath)
    print('saved the model to %s'%savePath)

# this method loads the saved model
def loadModel(sess, savedPath):
    saver = tf.train.Saver()
    saver.restore(sess, savedPath)
    print('loaded the model from to %s'%savedPath)

# this method returns the loss
def calcLoss(m, v, vTrue):
    # this is the absolute difference between the two tensors
    absDiff = tf.abs(m-vTrue)
    scale = 10
    v_sum = tf.reduce_sum(v) / scale
    vTrue_sum = tf.reduce_sum(vTrue) / scale

    maskZeros = vTrue
    maskOnes = 1 - vTrue

    # this is the error for not filling the voxels that are supposed to be filled
    error_ones = tf.reduce_sum(tf.mul(absDiff, maskZeros))
    # this is the error for filling the voxels that are not supposed to be filled
    error_zeros = tf.reduce_sum(tf.mul(absDiff, maskOnes))

    # this is the dynamic factor representing how much you care about which error
    factor = tf.nn.sigmoid(v_sum - vTrue_sum)

    loss = (factor*error_zeros) + ((1-factor)*error_ones)
    return loss

# now creating all the variables in the model
with tf.variable_scope('vars'):
    # naming convention
    # w is for weights and b is fo r biases
    # c is for convolutional, f is for fully connected, d is for deconvolutional
    wf1 = weightVariable([imgSize[0]*imgSize[1], 10240],'wf1')
    bf1 = biasVariable([10240],'bf1')
    wf2 = weightVariable([10240, 9216],'wf2')
    bf2 = biasVariable([9216], 'bf2')

    wc1  = weightVariable([5,5,4,16], 'wc1')
    bc1 = biasVariable([16], 'bc1')
    wc2 = weightVariable([5,5,16,32], 'wc2')
    bc2 = biasVariable([32], 'bc2')

    wd1 = weightVariable([5,5,5,16,32],'wd1')
    bd1 = biasVariable([16], 'bd1')
    wd2 = weightVariable([5,5,5,1,16],'wd2')
    bd2 = biasVariable([1], 'bd2')

# [-1, 96,128,1] - view
# [-1, 12288] - h_flat
# [-1, 10240] - h1
# [-1, 9216] - h2

# [-1, 48, 48, 4] - h2d
# [-1, 24, 24, 16] - h_conv1
# [-1, 12, 12, 32] - h_conv2

# [-1, 6,6,4,32] - m0
# [-1, 12, 12, 8, 16] - m1
# [-1, 24, 24, 16, 1] - m2

view = tf.placeholder(tf.float32, shape=[None, imgSize[0], imgSize[1], 1])
voxTrue = tf.placeholder(tf.float32, shape=[None, 24, 24, 16, 1])

h_flat = tf.reshape(view, [-1, imgSize[0]*imgSize[1]])
h1 = tf.nn.relu(tf.matmul(h_flat, wf1) + bf1)
h2 = tf.nn.relu(tf.matmul(h1, wf2) + bf2)

h2d = tf.reshape(h2, [-1,48,48,4])
h_conv1 = tf.nn.relu(conv2d(h2d, wc1) + bc1)
h_conv2 = tf.nn.relu(conv2d(h_conv1, wc2) + bc2)

m0 = tf.reshape(h_conv2, [-1,6,6,4,32])

m1 = tf.nn.relu(deConv3d(m0, wd1, [batch_size, 12,12,8,16]) + bd1)
m2 = tf.nn.sigmoid(deConv3d(m1, wd2, [batch_size, 24,24,16,1]) + bd2)

vox = tf.floor(2*m2)

loss = calcLoss(m2, vox, voxTrue)

optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)
accuracy = 100*tf.reduce_mean(tf.cast(tf.equal(vox, voxTrue), tf.float32))