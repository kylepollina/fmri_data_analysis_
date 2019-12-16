from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVM:
    def __init__(self, 
            training_samples, 
            training_labels, 
            test_samples, 
            test_labels, 
            pca_n_components, 
            cv_folds,
            decision_function_shape):
        self.X_train = training_samples
        self.y_train = training_labels
        self.X_test = test_samples
        self.y_test = test_labels
        self.pca = PCA(n_components=pca_n_components)
        self.cv_folds = cv_folds
        self.learning_function = SVC(gamma='scale', decision_function_shape=decision_function_shape)
        self.is_trained = False
        self.is_verbose = False

    def print(self, string):
        if(self.is_verbose):
            print(string)

    def train(self):
        self.print("Training svm...")

        self.X_train_pca = self.pca.fit_transform(self.X_train)
        self.X_test_pca  = self.pca.fit_transform(self.X_test)
        self.learning_function.fit(self.X_train_pca, self.y_train)
        self.is_trained = True

    def test(self):
        self.print("Testing svm...")

        predictions = self.learning_function.predict(self.X_test_pca)
        confusion_matrix = np.zeros((12, 12))

        for i in range(len(predictions)):
            predicted_label = predictions[i]
            correct_label = self.y_test[i]

            confusion_matrix[predicted_label][correct_label] += 1

        self.confusion_matrix = confusion_matrix
        self.predictions = predictions
        self.accuracy_score = accuracy_score(predictions, self.y_test)


