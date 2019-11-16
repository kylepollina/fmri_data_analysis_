import scipy.io
import numpy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

MAX_X = 51
MAX_Y = 61
MAX_Z = 23

def get_voxel_value(mat, trial, x, y, z):
    column = mat['meta']['coordToCol'][0][0][x][y][z]
    if column != 0:
        return mat['data'][trial][0][0][column-1]
    else:
        return 0

def is_voxel_valid(mat, x, y, z):
    return mat['meta']['coordToCol'][0][0][x][y][z] != 0

def run(data_file_name, trial, graph_file_name):
    mat = scipy.io.loadmat(data_file_name)

    fig = plt.figure()

    #add color mapper
    dataMin = mat['data'][trial][0][0].min()
    dataMax = mat['data'][trial][0][0].max()
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
