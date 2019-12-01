import preprocessing
import kfoldcv
import numpy as np
from visualize import visualize_KNN_k, visualize_precision_recall
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def svm():
    test_X, test_y, training_X, training_y = preprocessing.run()

    #Training
    clf = SVC(gamma='scale', decision_function_shape='ovo')
    pca = PCA(n_components = 100)

    pca_training_X = pca.fit_transform(training_X)
    pca_test_X = pca.transform(test_X)

    clf.fit(pca_training_X, training_y)

    #Testing
    predictions = clf.predict(pca_test_X)
    confusion_matrix = np.zeros((12, 12))
    for prediction, label in zip(predictions, test_y):
        confusion_matrix[prediction][label] += 1

    print('SVM Accuracy: ', accuracy_score(predictions, test_y))

    #Graphing
    visualize_precision_recall(confusion_matrix, 'SVM_Precision_Recall_Plot')

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

    #for random seed 30 when shuffling data
    train_error = np.array([0.910490152897502, 0.912516202101554, 0.9122242496696659, 
                            0.9104851192348832, 0.9084590700308312, 0.9145460265525702, 
                            0.9116474758279326, 0.9104880555380775, 0.9148295895467605, 
                            0.9130950733027119, 0.9130950733027119, 0.9130950733027119, 
                            0.9130950733027119, 0.9130950733027119, 0.9130950733027119, 
                            0.9119356530128568, 0.9165683005096582, 0.9165683005096582, 
                            0.9165683005096582, 0.9165683005096582])

    val_error = np.array([0.9235448312945289, 0.9282094367522516, 0.9258502486893398, 
                            0.9269995967199891, 0.9223820405968544, 0.9154187390778329, 
                            0.9096115069229735, 0.9211990858986423, 0.920002688533405, 
                            0.9188600618362683, 0.9235045032934532, 0.9188869471703185, 
                            0.9200430165344805, 0.9130864363489716, 0.9200699018685305, 
                            0.9247143433257158, 0.9235179459604786, 0.9339696195725231, 
                            0.9281892727517139, 0.9328001075413361])
    BEST_K = 7

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

    print('KNN Accuracy: ', accuracy_score(predictions, test_y))

    #Graphing
    visualize_KNN_k(train_error, val_error, 'KNN_K_Value_Plot')
    visualize_precision_recall(confusion_matrix, 'KNN_Precision_Recall_Plot')

svm()
#knn()