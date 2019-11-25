from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import numpy as np
num_folds = 5

# returns a vector of 5 mean squared errors
def svm5F(X, y):
    #line to change for the other model
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    pca = PCA(n_components = 100)
    n, d = X.shape
    ztrain = np.zeros((num_folds, 1))
    zval = np.zeros((num_folds, 1))
    for i in range(num_folds):
        #get indices of the current fold
        start_index = int(n * i / num_folds)
        end_index = int(n * (i + 1) / num_folds)
        T = set(range(start_index, end_index))
        S = set(range(n)) - T #set difference

        #fit the model
        Xfold = X[list(S)]
        yfold = y[list(S)]
        Xfold = pca.fit_transform(Xfold)

        clf.fit(Xfold, yfold)

        #for every incorrectly predicted point, increase the error for that point by 1
        zval[i] = [0]
        for t in T:
            if y[t] != clf.predict(pca.transform(X[t][None])):
                zval[i][0] += 1

        #then divide by the number of test points for a weighted error
        zval[i][0] /= len(T)

        ztrain[i] = [0]
        for s in S:
            if y[s] != clf.predict(pca.transform(X[t][None])):
                ztrain[i][0] += 1

        ztrain[i][0] /= len(S)

        return ztrain, zval

def knn5F(X, y, k):
    n, d = X.shape
    ztrain = np.zeros((num_folds, 1))
    zval = np.zeros((num_folds, 1))
    clf = KNeighborsClassifier(n_neighbors = k)
    pca = PCA(n_components = 100)

    for i in range(num_folds):
        #get indices of the current fold
        start_index = int(n * i / num_folds)
        end_index = int(n * (i + 1) / num_folds)
        T = set(range(start_index, end_index))
        S = set(range(n)) - T #set difference

        #fit the model
        Xfold = X[list(S)]
        yfold = y[list(S)]
        Xfold = pca.fit_transform(Xfold)
        clf.fit(Xfold, yfold)

        #for every incorrectly predicted point, increase the error for that point by 1
        zval[i] = [0]
        for t in T:
            if y[t] != clf.predict(pca.transform(X[t])):
                zval[i][0] += 1
        #then divide by the number of test points for a weighted error
        zval[t] /= len(T)

        ztrain[i] = [0]
        for s in S:
            if y[s] != clf.predict(pca.transform(X[t])):
                ztrain[i][0] += 1
        ztrain[i] /= len(S)

        return ztrain, zval
