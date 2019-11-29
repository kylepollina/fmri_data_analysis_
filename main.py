import preprocessing
import kfoldcv
import numpy as np
from visualize import visualize_KNN_k, visualize_precision_recall
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def svm():
    testX, testy, trainingX, trainingy = preprocessing.run()
    zSVM = kfoldcv.svm5F(trainingX, trainingy)

def knn():
    test_X, test_y, training_X, training_y = preprocessing.run()

    #Hyperparam Tuning
    """
    k_num_values = 20
    val_error = np.zeros(k_num_values)
    train_error = np.zeros(k_num_values)

    for k in range(k_num_values):
        #k values are (1 to 20), but are stored as (0 to 19)
        ztrain, zval = kfoldcv.knn5F(training_X, training_y, k+1)
        val_error[k] = zval.mean()
        train_error[k] = ztrain.mean()
    """

    train_error = np.array([0.91598859, 0.91685816, 0.91830743, 0.91772856, 0.91859812, 0.91888756,
                            0.91888756, 0.91975755, 0.91714885, 0.91714885, 0.91656914, 0.91656914,
                            0.91714885, 0.91656914, 0.91714885, 0.91511986, 0.91511986, 0.91511986,
                            0.91511986, 0.91309088])
    val_error = np.array([0.91539858, 0.92818255, 0.92237532, 0.9269996, 0.93281355, 0.92008334,
                            0.91544562, 0.91196397, 0.91539858, 0.91771744, 0.93394946, 0.93045436,
                            0.92466057, 0.92233499, 0.92119909, 0.93392929, 0.92927813, 0.92578976,
                            0.92930501, 0.92698615])
    BEST_K = 8

    #Training
    clf = KNeighborsClassifier(n_neighbors = BEST_K)
    pca = PCA(n_components = 100)

    pca_training_X = pca.fit_transform(training_X)
    pca_test_X = pca.transform(test_X)

    clf.fit(pca_training_X, training_y)

    #Testing
    predictions = clf.predict(pca_test_X)
    confusion_matrix = np.zeros((12, 12))
    for prediction, label in zip(predictions, test_y):
        confusion_matrix[prediction][label] += 1

    #Graphing
    visualize_KNN_k(train_error, val_error, 'KNN_K_Value_Plot')
    visualize_precision_recall(confusion_matrix, 'KNN_Precision_Recall_Plot')
