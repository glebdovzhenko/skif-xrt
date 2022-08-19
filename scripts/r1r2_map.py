import pickle

import numpy as np
import os

from matplotlib import pyplot as plt

from utils.xrtutils import get_line_kb, get_integral_breadth
from utils.various import datafiles


if __name__ == '__main__':
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/r1r2_map'

    r1s, r2s, alphas, ts, es, flux, z_breadth = [], [], [], [], [], [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'ES' or metadata['axes'] != 'XZ':
            continue

        print(metadata)

        r1s.append(metadata['r1'])
        r2s.append(metadata['r2'])
        alphas.append(metadata['alpha'])
        ts.append(metadata['thickness'])
        es.append(metadata['energy'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)
            z_breadth.append(get_integral_breadth(data, 'y'))

    r1s, r2s, flux, z_breadth, alphas, ts, es = np.array(r1s), np.array(r2s), np.array(flux), np.array(z_breadth), \
                                                np.array(alphas), np.array(ts), np.array(es)
    u_r1s, u_r2s = np.array(sorted(set(r1s))), np.array(sorted(set(r2s)))

    print(u_r1s, u_r2s)
    u_flux, u_z_breadth = np.nan + np.zeros(shape=(u_r1s.shape[0], u_r2s.shape[0])), \
                          np.nan + np.zeros(shape=(u_r1s.shape[0], u_r2s.shape[0]))

    for ii, r1 in enumerate(u_r1s):
        for jj, r2 in enumerate(u_r2s):
            u_flux[ii, jj] = flux[(r1s == r1) & (r2s == r2)].mean()
            u_z_breadth[ii, jj] = z_breadth[(r1s == r1) & (r2s == r2)].mean()

    plt.figure()
    for ii, r1 in enumerate(u_r1s):
        plt.semilogy((u_r2s - r1) * 1e3, u_flux[ii], label=r'$R_1 = %.01f$ m' % r1)

    plt.xlabel('$R_2 - R_1$, mm')
    plt.ylabel('Flux, [ph/s]')
    plt.legend()
    plt.tight_layout()

    # plt.figure()
    # for ii, r1 in enumerate(u_r1s):
    #     plt.plot((u_r2s - r1) * 1e3, u_z_breadth[ii], label='$R_1 = %.01f $' % r1)
    #
    # plt.xlabel('$R_2 - R_1$, mm')
    # plt.ylabel('Z integral breadth, [mm]')
    # plt.legend()
    # plt.tight_layout()
    plt.show()
