from model_0 import *

testSet = dataset('data/')
# ballDataset = dataset('ball_dataset/')
with tf.Session() as sess:
    loadModel(sess, model_save_path[0])

    batch = testSet.test_batch(batch_size*14)
    batch = testSet.test_batch(batch_size)
    # imgs = prepareImages(batch[0])
    newBatch = sess.run([view, vox], feed_dict={
        view: batch[0]
    })

    saveResults(newBatch, fileName='vox.pkl', saveImages=False)

    for i in range(batch_size):
        img = toImage(batch[0][i:i+1], imSize = [96,128])
        img.save('results/%s.png'%(i))