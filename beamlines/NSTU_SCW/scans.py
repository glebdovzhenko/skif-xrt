import os
import pickle
import shutil
from typing import List
import matplotlib
import numpy as np
import git
import csv

import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from NSTU_SCW import FL, NSTU_SCW
from components import PrismaticLens
from params.params_nstu_scw import (
    croc_crl_L,
    croc_crl_distance,
    croc_crl_y_t,
    diamond_filter_N,
    front_end_opening,
    monochromator_x_lim,
    monochromator_y_lim,
    sic_filter_N,
)
from utils.xrtutils import get_integral_breadth, get_line_kb, get_minmax

matplotlib.use('agg')


plots = []
x_kwds = {r'label': r'$x$', r'unit': r'mm', r'data': raycing.get_x}
y_kwds = {r'label': r'$y$', r'unit': r'mm', r'data': raycing.get_y}
z_kwds = {r'label': r'$z$', r'unit': r'mm', r'data': raycing.get_z}
xpr_kwds = {r'label': r'$x^{\prime}$',
            r'unit': r'', r'data': raycing.get_xprime}
zpr_kwds = {r'label': r'$z^{\prime}$',
            r'unit': r'', r'data': raycing.get_zprime}


for beam, t1 in zip(
        ('BeamAperture1Local', 'BeamMonoC1Local', 'BeamMonitor1Local',
         'BeamMonoC2Local', 'BeamMonitor2Local'),
        ('FE', 'C1', 'C1C2', 'C2', 'FM')):
    if t1 not in ('C1', 'C2'):
        params = zip(('XZ', 'XXpr', 'ZZpr'), (x_kwds, x_kwds,
                     z_kwds), (z_kwds, xpr_kwds, zpr_kwds))
    else:
        params = zip(('XY', 'XXpr'), (x_kwds, x_kwds), (y_kwds, xpr_kwds))

    for t2, xkw, ykw in params:
        plots.append(xrtplot.XYCPlot(beam=beam,
                                     title='-'.join((t1, t2)),
                                     xaxis=xrtplot.XYCAxis(**xkw),
                                     yaxis=xrtplot.XYCAxis(**ykw),
                                     aspect='auto'))


# @FL.gnrtr(50e3, 70e3, 20)
def onept(bl: NSTU_SCW, plts: List):
    subdir = os.path.join(os.getenv('BASE_DIR', ''), 'datasets', 'nstu-scw-2')
    scan_name = 'onept'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 30.e3
    if np.isclose(en, 30e3):
        r1, r2 = -2.04e3, -2.04e3  # 30 keV
        g_f = 1.076                # 30 keV
        d_en = 5e-3
    elif np.isclose(en, 50e3):
        r1, r2 = -1.22e3, -1.22e3  # 50 keV
        g_f = 0.390                # 50 keV
        d_en = 5e-3
    elif np.isclose(en, 70e3):
        r1, r2 = -.870e3, -.870e3  # 70 keV
        g_f = .191                 # 70 keV
        d_en = 1e-2
    elif np.isclose(en, 90e3):
        r1, r2 = -.675e3, -.675e3  # 90 keV
        g_f = .101                 # 90 keV
        d_en = 3e-2
    else:
        raise ValueError('En is not in [30, 50, 70, 90] keV')

    bl.align_source(en, d_en)
    bl.align_crl(croc_crl_L, int(croc_crl_L), croc_crl_y_t, g_f, 0.)
    bl.align_crl_mask(100., .2)
    bl.align_mono(en, r1, -6. * r1, r2, -6 * r2)

    for plot in plts:
        if 'FM' in plot.title:
            plot.saveName = os.path.join(
                subdir,
                scan_name,
                '%s.png' % (plot.title)
            )
            if 'XZ' in plot.title:
                plot.xaxis.limits = [-.5, .5]
                plot.yaxis.limits = [-.2, .2]

    r = git.Repo(os.getenv('BASE_DIR'))
    assert not r.is_dirty()
    assert r.head.ref == r.heads.rtr
    commit_name = r.head.commit.name_rev.replace(' rtr', '')
    metadata = bl._metadata.copy()
    metadata['commit'] = commit_name

    with open(os.path.join(subdir, scan_name, 'md.csv'), 'w') as ff:
        ff.write('\n'.join(('%s,%s' % (k, str(val)) for k, val in metadata.items())))

    yield


if __name__ == '__main__':
    beamline = NSTU_SCW()
    scan = onept
    show = False
    repeats = 1

    if show:
        beamline.glow(
            scale=[1e3, 1e3, 1e3],
            centerAt=r'Si[111] Crystal 1',
            generator=scan,
            generatorArgs=[beamline, plots],
            startFrom=1
        )
    else:
        xrtrun.run_ray_tracing(
            beamLine=beamline,
            plots=plots,
            repeats=repeats,
            backend=r"raycing",
            generator=scan,
            generatorArgs=[beamline, plots]
        )
