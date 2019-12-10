from sklearn import svm
import preprocessing
import kfoldcv

def run():
    X, y = preprocessing.run()
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    #for now, just run the function. eventually, we'll be iterating through these, passing in different tuning parameters each time
    z = kfoldcv.svm5F(X, y)
    return clf.fit(X, y)
