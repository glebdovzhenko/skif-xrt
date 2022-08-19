import pickle

import numpy as np
import os

from matplotlib import pyplot as plt

from utils.xrtutils import get_line_kb, get_integral_breadth
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

    rs, flux, z_breadth, alphas, ts, es = np.array(rs), np.array(flux), np.array(z_breadth), np.array(alphas), np.array(ts), np.array(es)
    # ii = np.argsort(rs)
    # ii = np.argsort(alphas)
    # ii = np.argsort(ts)
    ii = np.argsort(es)
    rs, flux, z_breadth, alphas, ts, es = rs[ii], flux[ii], z_breadth[ii], alphas[ii], ts[ii], es[ii]

    plt.figure()
    plt.semilogy(es, flux)
    plt.figure()
    plt.plot(es, z_breadth)
    plt.show()

    # flux_inf, z_breadth_inf = flux[np.isinf(rs)].mean(), z_breadth[np.isinf(rs)].mean()
    #
    # plt.figure()
    # plt.semilogy(rs, flux)
    # plt.plot([np.min(rs), np.ma.masked_invalid(rs).max()], [flux_inf, flux_inf], '--')
    # plt.figure()
    # plt.plot(rs, z_breadth)
    # plt.plot([np.min(rs), np.ma.masked_invalid(rs).max()], [z_breadth_inf, z_breadth_inf], '--')
    # plt.show()
