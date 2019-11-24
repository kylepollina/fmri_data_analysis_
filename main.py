import preprocessing
import kfoldcv

testX, testy, trainingX, trainingy = preprocessing.run()

#how to run:
zSVM = kfoldcv.svm5F(trainingX, trainingy)
zkNN = kfoldcv.knn5F(trainingX, trainingy, 3)
