from ops import *
import sys

# global params
batch_size = 5
learning_rate = 0.0001
model_save_path = 'savedModels/model_1.ckpt'
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

# now creating all the variables in the model
with tf.variable_scope('vars'):
    wc1 = weightVariable([5,5,1,24],'wc1')
    bc1 = biasVariable([24],'bc1')
    wc2 = weightVariable([5,5,24,48],'wc2')
    bc2 = biasVariable([48], 'bc2')

    wd1 = weightVariable([5,5,5,8,16],'wd1')
    bd1 = biasVariable([8], 'bd1')
    wd2 = weightVariable([5,5,5,1,8],'wd2')
    bd2 = biasVariable([1], 'bd2')

# [-1, 24,32,1]
# [-1, 12, 16, 24] - h1
# [-1, 6, 8, 48] - h2

# [-1, 6,6,4,16] - m0

# [-1, 12, 12, 8, 8] - m1
# [-1, 24, 24, 16, 1] - m2

view = tf.placeholder(tf.float32, shape=[None, 24, 32, 1])
voxTrue = tf.placeholder(tf.float32, shape=[None, 24, 24, 16, 1])

h1 = tf.nn.relu(conv2d(view, wc1) + bc1)
h2 = tf.nn.relu(conv2d(h1, wc2) + bc2)

m0 = tf.reshape(h2, [-1,6,6,4,16])

m1 = tf.nn.relu(deConv3d(m0, wd1, [batch_size, 12,12,8,8]) + bd1)
m2 = tf.nn.sigmoid(deConv3d(m1, wd2, [batch_size, 24,24,16,1]) + bd2)

loss = tf.reduce_mean(tf.square(m2 - voxTrue))

optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)

vox = tf.floor(2*m2)
accuracy = 100*tf.reduce_mean(tf.cast(tf.equal(vox, voxTrue), tf.float32))