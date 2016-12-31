from model import *

rhinoDataset = dataset('data/')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # loadModel(sess, model_save_path)

    cycles = 8000
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

        if i % 20 == 0:
            testBatch = rhinoDataset.test_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={
                view: testBatch[0],
                voxTrue: testBatch[1]
            })

            print('Accuracy: %.2f%s'%(acc, ' '*50))
    
    # now saving the trained model every 1500 cycles
        if i % 1000 == 0 and i != 0:
            saveModel(sess, model_save_path)
    
    # saving the model in the end
    saveModel(sess, model_save_path)