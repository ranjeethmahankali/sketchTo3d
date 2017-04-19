from ops import *
import sys

# this method returns the custom defined loss
def calcLoss(m, v, vTrue):
    # this is the absolute difference between the two tensors
    absDiff = tf.abs(m-vTrue)
    scale = 2
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

# this is for sigmoid cross entropy loss
def sigmoid_loss(m , vTrue):
    epsilon = 1e-9
    cross_entropy = -((vTrue * tf.log(m + epsilon))+((1-vTrue)*tf.log(1-m+epsilon)))
    
    # adding this to summaries
    ce_by_example = tf.reduce_sum(cross_entropy, axis=1, name='cross_entropy')
    summarize(ce_by_example)

    ce_loss = tf.reduce_sum(ce_by_example)

    # now implementing regularizations
    l2_loss = 0
    for v in varList:
        l2_loss += tf.nn.l2_loss(v)

    l2_loss *= alpha
    total_loss = ce_loss + l2_loss

    with tf.name_scope('loss_params'):
        tf.summary.scalar('l2_loss', l2_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss

# this returns the accuracy tensor
def accuracy(v, vTrue):
    correctness = tf.equal(v, vTrue)
    acc_norm = tf.cast(correctness, tf.float32)
    acc = tf.multiply(acc_norm, 100)

    acc_by_example = tf.reduce_mean(acc, axis=1,name='accuracy')

    summarize(acc_by_example)

    return tf.reduce_mean(acc_by_example)

# now creating all the variables in the model
with tf.variable_scope('vars'):
    # naming convention
    # w is for weights and b is fo r biases
    # c is for convolutional, f is for fully connected, d is for deconvolutional
    wf1 = weightVariable([imgSize[0]*imgSize[1], 8192],'wf1')
    bf1 = biasVariable([8192],'bf1')
    wf2 = weightVariable([8192, 9216],'wf2')
    bf2 = biasVariable([9216], 'bf2')

    wc1  = weightVariable([5,5,4,16], 'wc1')
    bc1 = biasVariable([16], 'bc1')
    wc2 = weightVariable([5,5,16,32], 'wc2')
    bc2 = biasVariable([32], 'bc2')

    wd1 = weightVariable([5,5,5,16,32],'wd1')
    bd1 = biasVariable([16], 'bd1')
    wd2 = weightVariable([5,5,5,1,16],'wd2')
    bd2 = biasVariable([1], 'bd2')

# list of vars we care about
all_vars = tf.trainable_variables()
varList = [v for v in all_vars if 'vars' in v.name]

# [-1, 96,128,1] - view
# [-1, 3072] - h_flat
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

m1 = tf.nn.tanh(deConv3d(m0, wd1, [batch_size, 12,12,8,16]) + bd1)
m2 = tf.nn.sigmoid(deConv3d(m1, wd2, [batch_size, 24,24,16,1]) + bd2, name='output')
summarize(m2)

vox = graph = tf.round(m2, name='voxels')
summarize(vox)

# loss = calcLoss(m2, vox, voxTrue)
loss = sigmoid_loss(m2, voxTrue)

optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)
accuracy = accuracy(vox, voxTrue)

# this is for the summaries during the training
merged = tf.summary.merge_all()