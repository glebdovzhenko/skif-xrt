import pickle

import numpy as np
import os

from matplotlib import pyplot as plt

from utils.various import datafiles


if __name__ == '__main__':
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/ae_scan'

    flux, offset = [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'ES' or metadata['axes'] != 'XZ':
            continue

        print(metadata)

        offset.append(metadata['tth_offset'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)

    flux, offset = np.array(flux), np.array(offset)
    ii = np.argsort(offset)
    flux, offset = flux[ii], offset[ii]

    plt.plot(offset, flux, '+-')
    plt.xlabel(r'$\theta - \theta_{B}$, [deg]')
    plt.ylabel(r'Flux, [ph/s]')
    plt.xlim(-.1, .1)
    plt.show()
