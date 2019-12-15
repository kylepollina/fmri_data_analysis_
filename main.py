from src.preprocessing import *

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pdb

training_samples, training_labels, test_samples, test_labels = preprocess_data(training_percentage=80)

def run():
    svm = build_svm()
    svm.train()
    svm.test()

    knn = build_knn()

def build_svm():
    return SVM(
            training_samples, 
            training_labels, 
            test_samples, 
            test_labels, 
            pca_n_components=100, 
            decision_function_shape='ovo')

def build_knn():
    return KNN(
            training_samples,
            training_labels,
            test_samples,
            test_labels,
            pca_n_components=10,
            k=1)

class KNN:
    def __init__(self,
            training_samples,
            training_labels,
            test_samples,
            test_labels,
            pca_n_components,
            k):
        self.X_train = training_samples
        self.y_train = training_labels
        self.X_test = test_samples
        self.y_test = test_labels
        self.pca = PCA(n_components = pca_n_components)
        self.k = k
        self.learning_function = KNeighborsClassifier(n_neighbors = self.k)
        self.is_trained = False
        self.is_verbose = False

    def run(self):
        self.is_verbose = True
        self.train()
        self.tune_k(1, 20)

    def print(self, string):
        if(self.is_verbose):
            print(string)

    def train(self):
        self.print("Training knn...")

        self.X_train_pca = self.pca.fit_transform(self.X_train)
        self.X_test_pca = self.pca.fit_transform(self.X_test)
        self.learning_function.fit(self.X_train_pca, self.y_train)
        self.is_trained = True

    def test(self):
        self.print(f"Testing knn with k = {self.k}...")

        predictions = self.learning_function.predict(self.X_test_pca)
        confusion_matrix = np.zeros((12, 12))

        for i in range(len(predictions)):
            predicted_label = predictions[i]
            correct_label = self.y_test[i]

            confusion_matrix[predicted_label][correct_label] += 1

        self.confusion_matrix = confusion_matrix
        self.predictions = predictions
        self.accuracy_score = accuracy_score(predictions, self.y_test)

    def tune_k(self, min_k, max_k):
        self.print("Tuning k...")

        scores = []

        for k in range(min_k, max_k):
            self.change_k(k)
            self.train()
            self.test()
            print(self.accuracy_score)
            scores.append(self.accuracy_score)

        max_accuracy = max(scores)
        best_k = scores.index(max_accuracy) + min_k

        print("Maximum accuracy: " + str(max_accuracy))
        print("Best k: " + str(best_k))

    def change_k(self, k):
        self.k = k
        self.learning_function = KNeighborsClassifier(n_neighbors = self.k)

class SVM:
    def __init__(self, 
            training_samples, 
            training_labels, 
            test_samples, 
            test_labels, 
            pca_n_components, 
            decision_function_shape):
        self.X_train = training_samples
        self.y_train = training_labels
        self.X_test = test_samples
        self.y_test = test_labels
        self.learning_function = SVC(gamma='scale', decision_function_shape=decision_function_shape)
        self.pca = PCA(n_components=pca_n_components)
        self.is_trained = False

    def train(self):
        print("Training svm...")
        self.X_train_pca = self.pca.fit_transform(self.X_train)
        self.X_test_pca  = self.pca.fit_transform(self.X_test)
        self.learning_function.fit(self.X_train_pca, self.y_train)
        self.is_trained = True

    def test(self):
        print("Testing svm...")
        predictions = self.learning_function.predict(self.X_test_pca)
        confusion_matrix = np.zeros((12, 12))

        for i in range(len(predictions)):
            predicted_label = predictions[i]
            correct_label = self.y_test[i]

            confusion_matrix[predicted_label][correct_label] += 1

        self.confusion_matrix = confusion_matrix
        self.predictions = predictions
        self.accuracy_score = accuracy_score(predictions, self.y_test)





#    #Training
#    clf = KNeighborsClassifier(n_neighbors = BEST_K)
#    pca = PCA(n_components = 100)

#    pca_training_X = pca.fit_transform(training_X)
#    pca_test_X = pca.transform(test_X)

#    clf.fit(pca_training_X, training_y)

#    #Testing
#    predictions = clf.predict(pca_test_X)
#    confusion_matrix = np.zeros((12, 12))
#    for prediction, label in zip(predictions, test_y):
#        confusion_matrix[prediction][label] += 1

#    print('KNN Accuracy: ', accuracy_score(predictions, test_y))

#    #Graphing
#    visualize_KNN_k(train_error, val_error, 'KNN_K_Value_Plot')
#    visualize_precision_recall(confusion_matrix, 'KNN_Precision_Recall_Plot', 'KNN Precision and Recall Values by Category')


#    #Graphing
#    visualize_precision_recall(confusion_matrix, 'SVM_Precision_Recall_Plot', 'SVM Precision and Recall Values by Category')
