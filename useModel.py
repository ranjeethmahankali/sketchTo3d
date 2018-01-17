from model_0 import *
# from model import *

testSet = dataset('data/')
# testSet = dataset('ball_dataset/')
# ballDataset = dataset('ball_dataset/')
with tf.Session() as sess:
    loadModel(sess, model_save_path[0])
    acc_all = []
    for _ in range(50):
        batch = testSet.test_batch(batch_size)
        # imgs = prepareImages(batch[0])
        # print(type(batch[1]))
        newBatch = sess.run([view, vox, accTensor], feed_dict={
            view: batch[0],
            voxTrue: batch[1]
        })

        # saveResults(newBatch[:2], fileName='vox.pkl', saveImages=False)
        acc = newBatch[2]
        acc_all.append(np.mean(acc))
    
    print("Accuracy: %.2f" % (sum(acc_all)/len(acc_all)))
    # for i in range(batch_size):
    #     # img = toImage(batch[0][i:i+1], imSize = [96,128])
    #     img = toImage(batch[0][i:i+1])
    #     img.save('results_ball/%s.png'%(i))