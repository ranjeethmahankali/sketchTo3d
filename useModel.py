from model import *

# testSet = dataset('results/')
ballDataset = dataset('ball_dataset/')
with tf.Session() as sess:
    loadModel(sess, model_save_path[1])

    batch = ballDataset.test_batch(batch_size*14)
    batch = ballDataset.test_batch(batch_size)
    # imgs = prepareImages(batch[0])
    newBatch = sess.run([view, vox], feed_dict={
        view: batch[0]
    })

    saveResults(newBatch, fileName='vox_2.pkl', saveImages=False)

    for i in range(batch_size):
        img = toImage(batch[0][i:i+1])
        img.save('results_ball/%s.png'%(i+5))