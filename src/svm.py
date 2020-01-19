from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from confusion_matrix import confusion_matrix

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
        self.istrained = False
        self.isverbose = True

    def print(self, string):
        if(self.isverbose):
            print(string)

    def isverbose(self):
        return self.isverbose

    def istrained(self):
        return self.istrained

    def train(self):
        self.print("Training svm...")

        self.X_train_pca = self.pca.fit_transform(self.X_train)
        self.X_test_pca  = self.pca.fit_transform(self.X_test)
        self.learning_function.fit(self.X_train_pca, self.y_train)
        self.istrained = True

    def test(self):
        self.print("Testing svm...")

        predictions = self.learning_function.predict(self.X_test_pca)
        self.confusion_matrix = confusion_matrix(12, 12, predictions, self.y_test)
        self.predictions = predictions
        self.accuracy_score = accuracy_score(predictions, self.y_test)


