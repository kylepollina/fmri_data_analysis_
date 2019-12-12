from scipy.io import loadmat
import numpy as np
import math
import random

def preprocess_data():
    preprocessed_data = [] 
    labels = []

    for i in range(1, 10):
        filename = 'data/p' + str(i)
        patient = PatientData(filename)
        num_trials = 360 

        for j in range(num_trials):
            image = patient.get_image(j)
            label = patient.get_label(j)

            preprocessed_data.append(image)
            labels.append(label)

    #TODO: average samples of same word, same person

    X, y = suffle_data(preprocessed_data, labels)

    trials = X.shape[0]
    split_index = int(trials / 5)

    training_samples = X[split_index + 1 :]
    training_labels  = y[split_index + 1 :]
    test_samples     = X[0 : split_index]
    test_labels      = y[0 : split_index]

    return training_samples, training_labels, test_samples, test_labels

# TODO redo
def suffle_data(preprocessed_data, labels):
    random.seed(30)

    together = list(zip(preprocessed_data, labels))
    random.shuffle(together)
    X, y = zip(*together)

    return np.array(X), np.array(y)


class PatientData:
    def __init__(self, filename):
        self.data   = loadmat(filename)
        self.meta   = self.data['meta'][0][0]
        self.info   = self.data['info'][0]
        self.images = self.data['data']
        self.noun_categories = [
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

    def get_raw_image(self, index):
        return self.images[index][0][0]

    def get_image(self, index):
        image = self.get_raw_image(index)
        max_image_size = 19750
        margin = (len(image) - max_image_size) / 2
        left_margin = math.floor(margin)
        right_margin = math.floor(margin) - math.ceil(margin)
        truncated_image = image[left_margin:][:(len(image) - right_margin)]

        return truncated_image

    def get_epoch(self, index):
        return self.info[index][4][0][0]

    def get_label(self, index):
        noun_category = self.get_noun_category(index)
        numeric_label = self.noun_categories.index(noun_category)
        return numeric_label

    def get_noun_category(self, index):
        noun_category = self.info[index][0][0]
        return noun_category


