import pickle
from PIL import Image
import numpy as np
from ops import *
import time

# global params
imgSize = (24, 32)
batch_size = 5

# converts data to image
def toImage(data):
    data = np.reshape(data, imgSize)
    newData = 255*data
    # converting new data into integer format to make it possible to export it as a bitmap
    # in this case converting it into 8 bit integer
    newData = newData.astype(np.uint8)
    return Image.fromarray(newData)

# this method loads the data file
with open('data/1.pkl','rb') as inp:
    dSet = pickle.load(inp)

data = [np.expand_dims(np.array(dSet[0]),3), np.expand_dims(np.array(dSet[1]), 4)]
counter = 0
def next_batch():
    dSize = data[0].shape[0]
    global counter
    print(counter)
    batch = [data[0][counter: counter+batch_size], data[1][counter: counter+batch_size]]
    counter = (counter+5)%dSize

    return batch
# print(data[0].shape, data[1].shape)

for i in range(50):
    img = toImage(data[0][i:i+1])
    img.save('imgs/%s.png'%i)
# print(data[0].shape, data[1].shape)

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

optim = tf.train.AdamOptimizer(0.001).minimize(loss)

vox = tf.floor(2*m2)
accuracy = 100*tf.reduce_mean(tf.cast(tf.equal(vox, voxTrue), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cycles = 100
    startTime = time.time()
    for i in range(cycles):
        batch = next_batch()
        _, val = sess.run([optim, accuracy], feed_dict={
            view: batch[0],
            voxTrue: batch[1]
        })

        print(val)
    
    timeElapsed = time.time() - startTime

    print('time: %.2f seconds'%timeElapsed)