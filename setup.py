from scipy.io import loadmat

import pdb

def p1():
    p1 = loadmat('p1')
    meta = p1['meta'][0][0]
    info = p1['info'][0]
    data = p1['data']

    output = []

    for i in range(360):
        image = data[i][0][0]
        
        cond        = info[i][0][0]
        cond_number = info[i][1][0][0]
        word        = info[i][2][0]
        word_number = info[i][3][0][0]
        epoch       = info[i][4][0][0]

        output.append({'cond': cond, 'cond_number': cond_number, 'word': word, 'word_number': word_number, 'epoch': epoch})
