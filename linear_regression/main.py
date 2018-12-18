import numpy as np
import csv
from random import shuffle
import linear_regression as lr

#read data from csv file
with open('Admission_Predict.csv','r') as csvfile:
    creader = csv.reader(csvfile,delimiter=',')
    datalist = []
    for row in creader:
        if row[0].startswith('Serial'):
            continue
        else:
            item_list = []
            for item in row:
                item_list.append(float(item))
            datalist.append(item_list)

train_data = np.zeros((300,8))
test_data = np.zeros((100,8))

#shuffle the order of the data
x = [i for i in range(400)] 
shuffle(x)

#generate training_dataset
for i in range(0,300):
    j = x[i]
    sample = datalist[j]
    for k in range(0,8):
        train_data[i,k] = sample[k+1]
        
#generate test_dataset
for i in range(300,400):
    j = x[i]
    sample = datalist[j]
    for k in range(0,8):
        test_data[i-300,k] = sample[k+1]

trainX = train_data[:,0:7]
trainY = train_data[:,7]
testX = test_data[:,0:7]
testY = test_data[:,7]

#now as we have train/test datasets, let's call the regression model
weights = lr.train(trainX,trainY)
prediction = lr.predict(testX, weights)
score = lr.test(testY, prediction)
print("score: ", score)

