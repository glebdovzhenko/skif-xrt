import pickle
from uncertainties import umath
import numpy as np
import os

from matplotlib import pyplot as plt

from utils.xrtutils import get_line_kb, get_integral_breadth
from utils.various import datafiles


if __name__ == '__main__':
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/r1_scan'

    rs, alphas, ts, flux, z_breadth, s_dist = [], [], [], [], [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'C1C2' or metadata['axes'] != 'ZZpr':
            continue

        print(metadata)

        rs.append(metadata['r1'])
        alphas.append(metadata['alpha'])
        ts.append(metadata['thickness'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)
            z_breadth.append(get_integral_breadth(data, 'y'))
            if rs[-1] < 50:
                k, b = get_line_kb(data)
            else:
                data.total2D = data.total2D[::-1]
                k, b = get_line_kb(data)
                data.total2D = data.total2D[::-1]
                k *= -1

            y, z = 33721.01537088534 - 1. / k, 12.5 - b / k  # in bl coordinates
            s_dist.append(umath.sqrt((y - 33500) ** 2 + (z - 0.) ** 2))  # relative to the crystal center

    rs, alphas, ts, flux, z_breadth, s_dist = np.array(rs), np.array(alphas), np.array(ts), np.array(flux), np.array(z_breadth), np.array(s_dist),
    ii = np.argsort(rs)
    rs, alphas, ts, flux, z_breadth, s_dist = rs[ii], alphas[ii], ts[ii], flux[ii], z_breadth[ii], s_dist[ii]
    print(rs)
    print(s_dist)
    # plt.plot(rs, flux)
    plt.plot(rs, [x.n for x in s_dist], 'o')
    plt.show()

