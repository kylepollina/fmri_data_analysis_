from scipy.io import loadmat
from sklearn.decomposition import PCA
import numpy as np

def run():
    p1 = loadmat('p1')
    meta = p1['meta'][0][0]
    info = p1['info'][0]
    data = p1['data']

    total_trials = 360 
    preprocessed_data = [] 
    n = 0

    for j in range(total_trials):
        image = get_image(data, j)

        epoch = get_epoch(info, j)

        if epoch == 0 or epoch == 1 or epoch == 2:
            preprocessed_data.append(image)
            n += 1

    X = preprocessed_data
    pca = PCA(n_components=100)
    pca.fit_transform(X)

def get_image(data, index):
    return data[index][0][0]

def get_epoch(info, index):
    return info[index][4][0][0]
