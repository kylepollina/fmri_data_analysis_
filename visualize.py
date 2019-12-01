import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

MAX_X = 51
MAX_Y = 61
MAX_Z = 23

NUM_CATEGORIES = 12

"""Functions to create fMRI data visualization"""
def get_voxel_value(mat, trial, x, y, z):
    column = mat['meta']['coordToCol'][0][0][x][y][z]
    if column != 0:
        return mat['data'][trial][0][0][column-1]
    else:
        return 0

def is_voxel_valid(mat, x, y, z):
    return mat['meta']['coordToCol'][0][0][x][y][z] != 0

def visualize_data(data_file_name, trial, graph_file_name):
    mat = scipy.io.loadmat(data_file_name)

    fig = plt.figure()

    #add color mapper
    #dataMin = mat['data'][trial][0][0].min()
    #dataMax = mat['data'][trial][0][0].max()
    dataMin = -4.342 #a select range for p1 and word 'corn' to make colors of the same value the same
    dataMax = 6.672

    norm = matplotlib.colors.Normalize(vmin=dataMin, vmax=dataMax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.bwr)

    #create subplots
    for z in range(MAX_Z):
        ax = fig.add_subplot(5,5,z+1)
        ax.set_xlim([0, MAX_X])
        ax.set_ylim([0, MAX_Y])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.set_title('Z = ' + str(z), fontsize=10)

        for x in range(MAX_X):
            for y in range(MAX_Y):
                if is_voxel_valid(mat, x, y, z):
                    ax.add_patch(matplotlib.patches.Rectangle((x,y), 1, 1, color=mapper.to_rgba(get_voxel_value(mat, trial, x, y, z))))
                else:
                    ax.add_patch(matplotlib.patches.Rectangle((x,y), 1, 1, color=(0.0, 0.0, 0.0)))

    #add colorbar
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.bwr, norm=plt.Normalize(vmin=dataMin, vmax=dataMax))
    sm._A = []
    fig.colorbar(sm, cax=cbar_ax)

    plt.subplots_adjust(hspace=0.4)

    plt.savefig(graph_file_name, bbox_inches='tight')

"""Functions to create the precision and recall plots"""
def get_precision(confusion_matrix, category):
    tp = float(confusion_matrix[category][category])
    tp_fp = float(confusion_matrix.sum(axis=1)[category])

    if tp_fp == 0.0:
        print('Divide by zero error on category', category)
        return 0.0
    else:
        return tp / tp_fp

def get_recall(confusion_matrix, category):
    tp = float(confusion_matrix[category][category])
    tp_fn = float(confusion_matrix.sum(axis=0)[category])
    
    if tp_fn == 0.0:
        print('Divide by zero error on category', category)
        return 0.0
    else:
        return tp / tp_fn

def visualize_precision_recall(confusion_matrix, graph_file_name):
    categories = [
        'Manmade',
        'Building',
        'Buildpart',
        'Tool',
        'Furniture',
        'Animal',
        'Kitchen',
        'Vehicle',
        'Insect',
        'Vegetable',
        'Bodypart',
        'Clothing'
    ]

    precision_values = [get_precision(confusion_matrix, category) for category in range(NUM_CATEGORIES)]
    recall_values = [get_recall(confusion_matrix, category) for category in range(NUM_CATEGORIES)]

    locations = np.arange(NUM_CATEGORIES)
    width = 0.35

    fig, ax = plt.subplots()
    precision_rects = ax.bar(locations, precision_values, width, color='r')
    recall_rects = ax.bar(locations + width, recall_values, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Values')
    #ax.set_ylim([0.0, 1.0])
    ax.set_title('Precision and Recall Values by Category')
    ax.set_xticks(locations + width / 2)
    ax.set_xticklabels(categories, rotation='vertical')

    lgd = ax.legend((precision_rects[0], recall_rects[0]), ('Precision', 'Recall'), loc='upper center')

    plt.savefig(graph_file_name, bbox_inches='tight')

"""Function to create the KNN k value plot"""
def visualize_KNN_k(training_error, validation_error, graph_file_name):
    x = np.arange(1, training_error.size+1)

    fig, ax = plt.subplots()
    ax.plot(x, training_error, 'bo-', label='Training Error')
    ax.plot(x, validation_error, 'ro-', label='Validation Error')

    ax.set_xticks(x)
    ax.set_xlabel('K Value')
    ax.set_ylabel('Error')
    ax.set_title('Error vs K Value')
    ax.legend(loc='upper center')

    plt.savefig(graph_file_name, bbox_inches='tight')