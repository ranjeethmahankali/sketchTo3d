import tensorflow as tf

cluster = tf.train.ClisterSpec({
    'local':['localhost:2222'],
    'worker':[
        '69.91.143.8:2222'
    ]
    })

server = tf.train.Server(cluster, job_name='local', task_index=0)
server.join()