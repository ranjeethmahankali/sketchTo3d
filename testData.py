from ops import *
dtSet = dataset('data/')

num = 20
batch = dtSet.next_batch(num)

for i in range(num):
    img = toImage(batch[0][i: i+1])
    img.save('results/%s.png'%i)