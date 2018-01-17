# from model_0 import *
from model import *

# testSet = dataset('data/')
testSet = dataset('ball_dataset/')
# ballDataset = dataset('ball_dataset/')
with tf.Session() as sess:
    loadModel(sess, model_save_path[1])

    batch = testSet.test_batch(batch_size*14)
    batch = testSet.test_batch(batch_size)
    # imgs = prepareImages(batch[0])
    newBatch = sess.run([view, vox, accuracy], feed_dict={
        view: batch[0]
    })

    saveResults(newBatch[:2], fileName='vox.pkl', saveImages=False)
    acc = newBatch[2]
    print("Accuracy: %.2f"%np.mean(acc))
    for i in range(batch_size):
        # img = toImage(batch[0][i:i+1], imSize = [96,128])
        img = toImage(batch[0][i:i+1])
        img.save('results_ball/%s.png'%(i))