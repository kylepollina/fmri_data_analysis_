from scipy.io import loadmat
from sklearn.decomposition import PCA
import numpy as np
import math
import random

import pdb

def run():
    preprocessed_data = [] 
    labels = []

    for i in range(1, 10):
        filename = 'p' + str(i)
        data = loadmat(filename)
        meta = data['meta'][0][0]
        info = data['info'][0]
        images = data['data']

        total_trials = 360 

        for j in range(total_trials):
            image = get_image(images, j)
            epoch = get_epoch(info, j)
            label = get_label(info, j)

            if epoch == 0 or epoch == 1 or epoch == 2:
                preprocessed_data.append(image)
                labels.append(label)

    #TODO: average samples of same word, same person

    X, y = suffle_data(preprocessed_data, labels)

    #split data 20-80
    n, d = X.shape
    split_index = int(n / 5)
    test_X = X[0 : split_index]
    test_y = y[0 : split_index]
    training_X = X[split_index + 1 :]
    training_y = y[split_index + 1 :]
    
    return test_X, test_y, training_X, training_y

def suffle_data(preprocessed_data, labels):
    random.seed(30)

    together = list(zip(preprocessed_data, labels))
    random.shuffle(together)
    X, y = zip(*together)

    return np.array(X), np.array(y)

def get_image(images, index):
    image = images[index][0][0]

    #Something to consider: maybe get voxels such that each feature in each sample
    #corresponds to the same brain region

    # truncate the image so each patient image is same length
    max_image_size = 19750
    # center image truncation
    margin = (len(image) - max_image_size) / 2
    truncated_image = image[math.floor(margin):][:(len(image)-math.floor(margin)-math.ceil(margin))]

    return truncated_image

def get_epoch(info, index):
    return info[index][4][0][0]

def get_label(info, index):
    categories = [
            'manmade',
            'building',
            'buildpart',
            'tool',
            'furniture',
            'animal',
            'kitchen',
            'vehicle',
            'insect',
            'vegetable',
            'bodypart',
            'clothing'
            ]
    category = info[index][0][0]
    label = categories.index(category)
    return label

