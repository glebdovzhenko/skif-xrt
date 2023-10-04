import re
import os
import numpy as np
import pandas as pd


def datafiles(dd):
    f_name_re0 = re.compile(r'.+.pickle$')
    f_name_re1 = re.compile(r'^(?P<name>[^-]+)-(?P<axes>[^-]+)-')
    f_name_re2 = re.compile(
        r'^(?P<value>[\d.-]+|inf|-inf)(?P<units>mm|m|keV|deg|ddeg|arcsec|dmcm)-?')

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
            'thickness': None,
            'tth_offset': None,
            'xyz_offset': None
        }

        while m is not None:
            f_name = f_name[m.end():]
            m = f_name_re2.match(f_name)

            if m is None:
                break

            if m.group('units') == 'keV':
                result['energy'] = float(m.group('value'))
            elif m.group('units') == 'm':
                if result['r1'] is None:
                    result['r1'] = float(m.group('value'))
                elif result['r2'] is None:
                    result['r2'] = float(m.group('value'))
            elif m.group('units') == 'mm':
                result['thickness'] = float(m.group('value'))
            elif m.group('units') == 'deg':
                result['alpha'] = float(m.group('value'))
            elif m.group('units') == 'ddeg':
                result['tth_offset'] = float(m.group('value'))
            elif m.group('units') == 'arcsec':
                result['tth_offset'] = float(m.group('value'))
            elif m.group('units') == 'dmcm':
                result['xyz_offset'] = float(m.group('value'))

        yield result


def datafiles2(dd: str):
    assert os.path.exists(dd)
    assert os.path.exists(os.path.join(dd, 'md.csv'))

    # reading metadata common for all files
    md_base = pd.read_csv(os.path.join(dd, 'md.csv'), header=None, index_col=0)
    md_base = md_base.squeeze().to_dict()
    print(md_base)

    r0 = re.compile(r'^(?P<name>[^-]+)-(?P<axes>[^-]+)')
    r1 = re.compile(r'-(?P<key>{})-(?P<value>-?[\d.]+|inf|-inf)(.pickle$)?'.format('|'.join(md_base.keys())))

    for f_name in os.listdir(dd):
        mtc = r0.match(f_name)
        if mtc is None:
            continue

        result = md_base.copy()
        result['file'] = f_name

        pos = mtc.end()
        while True:
            mtc = r1.match(f_name[pos:])
            if mtc is None:
                break
            pos += mtc.end()
            gd = mtc.groupdict()
            result[gd['key']] = gd['value']
        
        yield result


if __name__ == '__main__':
    # dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/t_scan'

    # for data in datafiles(dd):
    #     print(data)

    dd = os.path.join(os.getenv('BASE_DIR'), 'datasets',
                      'nstu-scw-2', 'scan_mask_opening_30')

    for data in datafiles2(dd):
        print(data)
    dd = os.path.join(os.getenv('BASE_DIR'), 'datasets',
                      'nstu-scw-2', 'scan_lens_scale')
    for data in datafiles2(dd):
        print(data)
