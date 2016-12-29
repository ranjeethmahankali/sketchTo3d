import pickle
from PIL import Image
import numpy as np
from ops import *
# import time
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

# print(m0.get_shape(), wd1.get_shape())

m1 = tf.nn.relu(deConv3d(m0, wd1, [batch_size, 12,12,8,8]) + bd1)
m2 = tf.nn.sigmoid(deConv3d(m1, wd2, [batch_size, 24,24,16,1]) + bd2)

loss = tf.reduce_mean(tf.square(voxTrue - m2))

optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)

vox = tf.floor(2*m2)
accuracy = 100*tf.reduce_mean(tf.cast(tf.equal(vox, voxTrue), tf.float32))

rhinoDataset = dataset('data/')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # loadModel(sess, model_save_path)

    cycles = 2000
    startTime = time.time()

    for i in range(cycles):
        batch = rhinoDataset.next_batch(batch_size)
        _ = sess.run(optim, feed_dict={
            view: batch[0],
            voxTrue: batch[1]
        })

        timer = estimate_time(startTime, cycles, i)
        pL = 10 # this is the length of the progress bar to be displayed
        pNum = i % pL
        pBar = '#'*pNum + ' '*(pL - pNum)

        sys.stdout.write('...Training...|%s|-(%s/%s)- %s\r'%(pBar, i, cycles, timer))

        if i % 10 == 0:
            testBatch = rhinoDataset.test_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={
                view: testBatch[0],
                voxTrue: testBatch[1]
            })

            print('Accuracy: %.2f%s'%(acc, ' '*50))
    
    # now saving the trained model
    saveModel(sess, model_save_path)