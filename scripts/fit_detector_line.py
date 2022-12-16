import pickle
import os
from numpy.random import random_sample
from scipy.optimize import minimize
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from utils.xrtutils import get_line_kb, get_integral_breadth
from utils.various import datafiles


if __name__ == '__main__':
    # scan3 meridional +TT
    # scan2 sagittal no TT
    # scan4 sagittal +TT
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/PLAYGROUND/img/scan5'

    
    rs, f_dist, flux, br, br2 = [], [], [], [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'C2' or metadata['axes'] != 'XXpr':
            continue

        rs.append(metadata['r1'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)
            k, b = get_line_kb(data, show=False)
            f_dist.append(-np.sign(k) * np.sqrt((1. / k) ** 2 + (b / k) ** 2))
            br.append(get_integral_breadth(data, axis='y'))
            br2.append(get_integral_breadth(data, axis='x'))
            print('R = %f' % metadata['r1'])
            print('Focus F = %f, 2Θ = %f' % (f_dist[-1], np.degrees(np.arctan(b))))

    rs, f_dist, flux, br, br2 = np.array(rs), np.array(f_dist), np.array(flux), np.array(br), np.array(br2)
    ii = np.argsort(rs)
    rs, f_dist, flux, br, br2 = rs[ii], f_dist[ii] * 1e-3, flux[ii], br[ii], br2[ii]
    
    # f_dist -= np.interp(0., rs, f_dist)
    print(rs)
    print(f_dist)
    df = pd.DataFrame(f_dist, index=rs)
    print(df[(df.index >= 2.) & (df.index <= 6.)])

    fig, ax = plt.subplots()
    ax.plot(rs, br2, marker='.', linestyle='-', label='Размер пятна', color='C0')
    # ax.plot([rs[1], rs[-2]], [f_dist[-1], f_dist[-1]], '--', color='C0', label=r'Cr-to-F $R=\infty$')
    # ax.plot([rs[1], rs[-2]], [f_dist[0], f_dist[0]], '--', color='C0', label=r'Cr-to-F $R=-\infty$')
    plt.legend(loc='lower right')

    secax = ax.twinx()
    # secax.plot(rs, br, color='C1', label='Z\' Breadth')
    secax.plot(rs, flux, marker='.', linestyle='-', color='C1', label='Поток')
    plt.legend(loc='upper left')
    
    ax.set_ylabel('Размер пятна, мм')
    secax.set_ylabel('Поток, ф/с')   
    ax.set_xlabel('Радиус изгиба, м')

    plt.tight_layout()
    plt.show()

