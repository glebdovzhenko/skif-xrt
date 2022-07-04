import pickle
import re

import numpy as np
import os

from matplotlib import pyplot as plt


if __name__ == '__main__':
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/R1-R2-map-mag'
    step = 1.
    f_name_re = re.compile(r'[^-]+-(?P<energy>[\d.-]+)keV-(?P<r1>[\d.-]+)m-(?P<r2>[\d.-]+)m.pickle')

    r1s, r2s = set(), set()
    for f_name in filter(lambda x: ('DCM Slit Directions-' in x) and ('.pickle' in x), os.listdir(dd)):
        m = f_name_re.match(f_name)
        if m is None:
            continue

        r1s.add(m.group('r1'))
        r2s.add(m.group('r2'))

    r1s, r2s = {k: i for (k, i) in zip(sorted(r1s), range((len(r1s))))}, \
               {k: i for (k, i) in zip(sorted(r2s), range((len(r2s))))}

    flux = np.zeros(shape=(len(r1s), len(r2s))) + np.nan

    for f_name in filter(lambda x: ('DCM Slit Directions-' in x) and ('.pickle' in x), os.listdir(dd)):
        m = f_name_re.match(f_name)
        if m is None:
            continue

        with open(os.path.join(dd, f_name), 'rb') as f:
            data = pickle.load(f)
            flux[r1s[m.group('r1')], r2s[m.group('r2')]] = data.flux

    r1s, r2s = [float(k) for k in r1s.keys()], [float(k) for k in r2s.keys()]

    plt.imshow(flux, origin='lower',
               extent=[min(r2s) - .5 * step, max(r2s) + .5 * step,
                       min(r1s) - .5 * step, max(r1s) + .5 * step])
    plt.colorbar()
    plt.xlabel('$R_2$, м')
    plt.ylabel('$R_1$, м')
    plt.show()
