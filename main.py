import preprocessing
import kfoldcv

import pdb

def svm():
    testX, testy, trainingX, trainingy = preprocessing.run()
    zSVM = kfoldcv.svm5F(trainingX, trainingy)

def kNN():
    pdb.set_trace()
    testX, testy, trainingX, trainingy = preprocessing.run()
    k = 3
    zkNN = kfoldcv.knn5F(trainingX, trainingy, k)

