import numpy as np
import json
import os
import pickle
import xrt.backends.raycing.sources as rsources
from scipy.optimize import minimize
import params

from matplotlib import pyplot as plt
from utils.various import datafiles
from utils.xrtutils import get_integral_breadth, bell_fit, get_line_kb


if __name__ == '__main__':
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15/get_focus_s'
    for metadata in datafiles(dd):
        if (metadata['name'] == 'C1C2') & (metadata['axes'] == 'XXpr'):
            with open(os.path.join(dd, metadata['file']), 'rb') as f:
                data = pickle.load(f)
                k, b = get_line_kb(data, show=True)
                fdist = -np.sign(k) * np.sqrt((1. / k) ** 2 + (b / k) ** 2)
                print(metadata['r1'], fdist)
