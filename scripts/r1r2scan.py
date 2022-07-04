import pickle
import re

import numpy as np
import os

from matplotlib import pyplot as plt


if __name__ == '__main__':
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/R1-R2-scan'
    step = 1.
    f_name_re = re.compile(r'[^-]+-(?P<energy>[\d.-]+)keV-(?P<r1>[\d.-]+)m-(?P<r2>[\d.-]+)m.pickle')

    rs, flux = [], []
    for f_name in filter(lambda x: ('DCM Slit Directions-' in x) and ('.pickle' in x), os.listdir(dd)):
        m = f_name_re.match(f_name)
        if m is None:
            continue

        if m.group('r1') != m.group('r2'):
            continue

        rs.append(float(m.group('r1')))

        with open(os.path.join(dd, f_name), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)

    rs, flux = np.array(rs), np.array(flux)
    ii = np.argsort(rs)
    rs, flux = rs[ii], flux[ii]

    plt.plot(rs, flux)
    plt.xlabel('$R_1 = R_2$, м')
    plt.ylabel('$\Phi$, ф/с')
    plt.show()
