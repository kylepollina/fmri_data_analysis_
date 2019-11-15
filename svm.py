from sklearn import svm
import preprocessing

def run():
    X, y = preprocessing.run()
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    return clf.fit(X, y)
