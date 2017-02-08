from model import *

# testSet = dataset('results/')
with tf.Session() as sess:
    loadModel(sess, model_save_path)

    # batch = testSet.test_batch(batch_size)
    batch = prepareImages(['results/8.png']*batch_size)
    newBatch = sess.run([view, vox], feed_dict={
        # view: batch[0]
        view: batch
    })

    saveResults(newBatch, fileName='8.pkl', saveImages=False)