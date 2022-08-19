import re
import os
import numpy as np


def datafiles(dd):
    f_name_re0 = re.compile(r'.+.pickle$')
    f_name_re1 = re.compile(r'^(?P<name>[^-]+)-(?P<axes>[^-]+)-')
    f_name_re2 = re.compile(r'^(?P<value>[\d.-]+|inf)(?P<units>mm|m|keV|deg)-?')

    for f_name in os.listdir(dd):
        m0 = f_name_re0.match(f_name)
        m = f_name_re1.match(f_name)
        if m0 is None or m is None:
            continue

        result = {
            'file': f_name,
            'name': m.group('name'),
            'axes': m.group('axes'),
            'energy': None,
            'alpha': None,
            'r1': None,
            'r2': None,
            'thickness': None
        }

        while m is not None:
            f_name = f_name[m.end():]
            m = f_name_re2.match(f_name)

            if m is None:
                break

            if m.group('units') == 'keV':
                result['energy'] = np.float(m.group('value'))
            elif m.group('units') == 'm':
                if result['r1'] is None:
                    result['r1'] = np.float(m.group('value'))
                elif result['r2'] is None:
                    result['r2'] = np.float(m.group('value'))
            elif m.group('units') == 'mm':
                result['thickness'] = np.float(m.group('value'))
            elif m.group('units') == 'deg':
                result['alpha'] = np.float(m.group('value'))

        yield result


if __name__ == '__main__':
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/t_scan'

    for data in datafiles(dd):
        print(data)