import pickle
import os
from numpy.random import random_sample
from scipy.optimize import minimize
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from utils.xrtutils import get_integral_breadth
from utils.various import datafiles


if __name__ == '__main__':
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/de_to_e_br'

    
    rs, de, dzpr = [], [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'C1C2' or metadata['axes'] != 'ZZpr':
            continue

        rs.append(metadata['r1'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            de.append(get_integral_breadth(data, axis='e'))
            dzpr.append(get_integral_breadth(data, axis='y'))

    rs, de, dzpr = np.array(rs), np.array(de), np.array(dzpr)
    ii = np.argsort(rs)
    rs, de, dzpr = rs[ii], de[ii], dzpr[ii]
    
    plt.figure()
    plt.plot(rs, de)
    plt.figure()
    plt.plot(rs, dzpr)
    plt.show()
