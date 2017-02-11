from model import *

# testSet = dataset('results/')
with tf.Session() as sess:
    loadModel(sess, model_save_path[1])

    # batch = testSet.test_batch(batch_size)
    batch = prepareImages(['ball_dataset/8.png']*batch_size)
    newBatch = sess.run([view, vox], feed_dict={
        # view: batch[0]
        view: batch
    })

    saveResults(newBatch, fileName='vox.pkl', saveImages=False)