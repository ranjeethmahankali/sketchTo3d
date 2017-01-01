from model import *

testSet = dataset('results/')
with tf.Session() as sess:
    loadModel(sess, model_save_path)

    batch = testSet.test_batch(batch_size)
    newBatch = sess.run([view, vox], feed_dict={
        view: batch[0]
    })

    saveResults(newBatch, saveImages=False)