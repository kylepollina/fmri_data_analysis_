import preprocessing
import kfoldcv

import pdb

def svm():
    testX, testy, trainingX, trainingy = preprocessing.run()
    zSVM = kfoldcv.svm5F(trainingX, trainingy)

def knn():
    test_X, test_y, training_X, training_y = preprocessing.run()
    errors = []

    k_min = 1
    k_max = 20
    for k in range(k_min, k_max):
        z = kfoldcv.knn5F(trainingX, trainingy, k)
        errors.append({"k": k, "z": z})

        if(errors[len(errors) - 2].z < z):
            break

