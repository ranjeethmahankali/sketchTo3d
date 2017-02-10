from ops import *
dtSet = dataset('ball_dataset/')

num = 10
batch = dtSet.next_batch(num)

for i in range(num):
    img = toImage(batch[0][i: i+1])
    img.save('ball_dataset/%s.png'%i)