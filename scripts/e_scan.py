import pickle

import numpy as np
import os

from matplotlib import pyplot as plt

from utils.xrtutils import get_integral_breadth
from utils.various import datafiles


if __name__ == '__main__':
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/e_scan'

    rs, alphas, ts, es, flux, z_breadth = [], [], [], [], [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'C1C2' or metadata['axes'] != 'ZE':
            continue

        print(metadata)

        rs.append(metadata['r1'])
        alphas.append(metadata['alpha'])
        ts.append(metadata['thickness'])
        es.append(metadata['energy'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)
            z_breadth.append(get_integral_breadth(data, 'y'))

    rs, flux, z_breadth, alphas, ts, es = np.array(rs), np.array(flux), np.array(z_breadth), \
            np.array(alphas), np.array(ts), np.array(es)
    u_alphas, u_es = np.array(sorted(set(alphas))), np.array(sorted(set(es)))

    print(u_alphas, u_es)
    u_flux, u_z_breadth = np.nan + np.zeros(shape=(u_es.shape[0], u_alphas.shape[0])), \
                          np.nan + np.zeros(shape=(u_es.shape[0], u_alphas.shape[0]))

    for ii, alpha in enumerate(u_alphas):
        for jj, e in enumerate(u_es):
            u_flux[jj, ii] = flux[(alphas == alpha) & (es == e)].mean()
            u_z_breadth[jj, ii] = z_breadth[(alphas == alpha) & (es == e)].mean()

    for ii, alp in enumerate(u_alphas):
        plt.semilogy(u_es, u_flux[:, ii], label=r'$\chi = %.01f ^{\circ}$' % alp)

    plt.legend()
    plt.show()
