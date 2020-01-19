from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from confusion_matrix import confusion_matrix

class KNN:
    def __init__(self,
            training_samples,
            training_labels,
            test_samples,
            test_labels,
            pca_n_components,
            k,
            cv_folds):
        self.X_train = training_samples
        self.y_train = training_labels
        self.X_test = test_samples
        self.y_test = test_labels
        self.pca = PCA(n_components = pca_n_components)
        self.k = k
        self.cv_folds = cv_folds
        self.learning_function = KNeighborsClassifier(n_neighbors = self.k)
        self.istrained = False
        self.isverbose = True

    def run(self):
        self.isverbose = True
        self.train()
        scores, k_values = self.tune_k(1, 20)

    def print(self, string):
        if(self.isverbose):
            print(string)

    def isverbose(self):
        return self.isverbose

    def istrained(self):
        return self.istrained

    def train(self):
        self.print("Training knn...")

        self.X_train_pca = self.pca.fit_transform(self.X_train)
        self.X_test_pca = self.pca.fit_transform(self.X_test)
        self.learning_function.fit(self.X_train_pca, self.y_train)
        self.istrained = True

    def test(self):
        self.print(f"Testing knn with k = {self.k}...")

        predictions = self.learning_function.predict(self.X_test_pca)
        self.confusion_matrix = confusion_matrix(12, 12, predictions, self.y_test)
        self.predictions = predictions
        self.accuracy_score = accuracy_score(predictions, self.y_test)

    def tune_k(self, min_k, max_k):
        self.print("Tuning k...")

        scores = []

        for k in range(min_k, max_k):
            self.change_k(k)
            self.train()
            self.test()
            self.print(f"k = {k}    accuracy: {self.accuracy_score}")
            scores.append(self.accuracy_score)

        max_accuracy = max(scores)
        best_k = scores.index(max_accuracy) + min_k

        self.print(f"Highest Accuracy: {max_accuracy}")
        self.print(f"Best k: {best_k}")

        return scores, list(range(min_k, max_k))

    def change_k(self, k):
        self.k = k
        self.learning_function = KNeighborsClassifier(n_neighbors = self.k)
