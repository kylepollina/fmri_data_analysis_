from src.preprocessing import *
from src.svm import SVM
from src.knn import KNN

training_samples, training_labels, test_samples, test_labels = preprocess_data(training_percentage=80)

def build_svm():
    return SVM(
            training_samples, 
            training_labels, 
            test_samples, 
            test_labels, 
            pca_n_components=500, 
            cv_folds=5,
            decision_function_shape='ovo')

def build_knn():
    return KNN(
            training_samples,
            training_labels,
            test_samples,
            test_labels,
            pca_n_components=500,
            cv_folds=5,
            k=4)

# TODO implement cross validation
# TODO make pca_n_components an optional parameter
