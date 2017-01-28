from ops import *
dtSet = dataset('newData/')

batch = dtSet.next_batch(5)

for i in range(5):
    img = toImage(batch[0][i: i+1])
    img.save('data/%s.png'%i)