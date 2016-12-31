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
    w1 = weightVariable([768, 1536],'w1')
    b1 = biasVariable([1536],'b1')
    w2 = weightVariable([1536, 2304],'w2')
    b2 = biasVariable([2304], 'b2')

    wd1 = weightVariable([5,5,5,8,16],'wd1')
    bd1 = biasVariable([8], 'bd1')
    wd2 = weightVariable([5,5,5,1,8],'wd2')
    bd2 = biasVariable([1], 'bd2')

# [-1, 24,32,1] - view
# [-1, 768] - flattened
# [-1, 1536] - h1
# [-1, 2304] - h2

# [-1, 6,6,4,16] - m0

# [-1, 12, 12, 8, 8] - m1
# [-1, 24, 24, 16, 1] - m2

view = tf.placeholder(tf.float32, shape=[None, 24, 32, 1])
voxTrue = tf.placeholder(tf.float32, shape=[None, 24, 24, 16, 1])

flattened = tf.reshape(view, [-1, 768])

h1 = tf.nn.relu(tf.matmul(flattened, w1) + b1)
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

m0 = tf.reshape(h2, [-1,6,6,4,16])

m1 = tf.nn.relu(deConv3d(m0, wd1, [batch_size, 12,12,8,8]) + bd1)
m2 = tf.nn.relu(deConv3d(m1, wd2, [batch_size, 24,24,16,1]) + bd2)

vox = tf.floor(2*m2)

loss = tf.reduce_mean(tf.abs(m2 - voxTrue))
loss += tf.abs(tf.reduce_sum(voxTrue) - tf.reduce_sum(vox))

optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)
accuracy = 100*tf.reduce_mean(tf.cast(tf.equal(vox, voxTrue), tf.float32))