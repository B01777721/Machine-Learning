import numpy as np
from collections import namedtuple
import gzip
import logistic_regression as lo


f1 = open("data/train-images-idx3-ubyte.gz",'rb')
f2 = open("data/train-labels-idx1-ubyte.gz",'rb')
buf1 = gzip.GzipFile(fileobj=f1).read(16+60000*28*28)
buf2 = gzip.GzipFile(fileobj=f2).read(8+60000)

train_images = np.frombuffer(buf1,dtype='uint8',offset=16).reshape(60000,28*28)
train_images = np.where(train_images > 99, 1, 0)
train_labels = np.frombuffer(buf2,dtype='uint8',offset=8)


f3 = open("data/t10k-images-idx3-ubyte.gz",'rb')
f4 = open("data/t10k-labels-idx1-ubyte.gz",'rb')
buf3 = gzip.GzipFile(fileobj=f3).read(16+10000*28*28)
buf4 = gzip.GzipFile(fileobj=f4).read(8+10000)
test_images = np.frombuffer(buf3,dtype='uint8',offset=16).reshape(10000,28*28)
test_images = np.where(test_images > 99, 1, 0)
test_labels = np.frombuffer(buf4, dtype='uint8',offset=8)

#softmax regression training model 
weights = lo.train(train_images, train_labels,
                   numFeatures=784, numClasses=10, learning_rate=0.1)
prediction = lo.predict(test_images, weights)
score = lo.test(prediction,test_labels)

print("score is: ", score)
