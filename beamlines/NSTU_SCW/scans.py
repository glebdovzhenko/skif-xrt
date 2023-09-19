from typing import List
import os
import numpy as np
import matplotlib

matplotlib.use('agg')

import xrt.runner as xrtrun
import xrt.plotter as xrtplot
import xrt.backends.raycing as raycing

from utils.xrtutils import get_minmax, get_line_kb, get_integral_breadth
from components import PrismaticLens

from NSTU_SCW import NSTU_SCW
from params.params_nstu_scw import croc_crl_L, sic_filter_N, diamond_filter_N, front_end_opening, croc_crl_y_t, \
croc_crl_distance, croc_crl_L, monochromator_x_lim, monochromator_y_lim


plots = []
x_kwds = {r'label': r'$x$', r'unit': r'mm', r'data': raycing.get_x}
y_kwds = {r'label': r'$y$', r'unit': r'mm', r'data': raycing.get_y}
z_kwds = {r'label': r'$z$', r'unit': r'mm', r'data': raycing.get_z}
xpr_kwds = {r'label': r'$x^{\prime}$', r'unit': r'', r'data': raycing.get_xprime}
zpr_kwds = {r'label': r'$z^{\prime}$', r'unit': r'', r'data': raycing.get_zprime}


for beam, t1 in zip(('BeamAperture1Local', 'BeamMonoC1Local', 'BeamMonitor1Local', 'BeamMonoC2Local', 
                     'BeamMonitor2Local'), 
                    ('FE', 'C1', 'C1C2', 'C2', 'FM')):
    if t1 not in ('C1', 'C2'):
        params = zip(('XZ', 'XXpr', 'ZZpr'), (x_kwds, x_kwds, z_kwds), (z_kwds, xpr_kwds, zpr_kwds))
    else:
        params = zip(('XY', 'XXpr'), (x_kwds, x_kwds), (y_kwds, xpr_kwds))

    for t2, xkw, ykw in params:
        plots.append(xrtplot.XYCPlot(beam=beam, title='-'.join((t1, t2)), 
                                     xaxis=xrtplot.XYCAxis(**xkw), yaxis=xrtplot.XYCAxis(**ykw),
                                     aspect='auto'))


def onept(plts: List, bl: NSTU_SCW):
    subdir = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'nstu-scw')
    scan_name = 'spot_size'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 30.e3
    r1, r2 = -2.04, -2.04  # 30 keV
    # r1, r2 = -1.22, -1.22  # 50 keV
    # r1, r2 = -.87, -.87  # 70 keV
    # r1, r2 = -.675, -.675  # 90 keV
    bl.MonochromatorCr1.Rx = r1 * 1e3
    bl.MonochromatorCr1.Ry = -r1 * 6e3

    bl.MonochromatorCr2.Rx = r2 * 1.e3
    bl.MonochromatorCr2.Ry = -r2 * 6.e3

    bl.align_energy(en, 5e-2, invert_croc=False)
    # bl.SuperCWiggler.eMin = 100.
    # bl.SuperCWiggler.eMax = 1.e6

    del bl.CrocLensStack[:]
    g_f = 1.228  # 30 keV
    # g_f = .435  # 50 keV
    # g_f = .224  # 70 keV
    # g_f = .138  # 90 keV
    bl.CrocLensStack = PrismaticLens.make_stack(
        L=croc_crl_L, N=int(croc_crl_L), d=croc_crl_y_t, g_last=0., g_first=g_f,
        bl=bl, 
        center=[0., croc_crl_distance, 0],
        material=bl.LensMaterial,
        limPhysX=monochromator_x_lim, 
        limPhysY=monochromator_y_lim, 
    )

    bl.CrlMask.opening = [-100, 100, -100, 100]

    for plot in plts:
        if plot.title == 'FM-XZ':
            plot.xaxis.limits = [-.5, .5]
            plot.yaxis.limits = [-.5, .5]
            plot.caxis.limits = [bl.SuperCWiggler.eMin, bl.SuperCWiggler.eMax]
            plot.saveName = os.path.join(subdir, scan_name, '%s-%dkeV.png' % (plot.title, int(en * 1e-3)))
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

    yield


if __name__ == '__main__':
    beamline = NSTU_SCW()
    scan = onept
    show = False
    repeats = 20

    if show:
        beamline.glow(
            scale=[1e3, 1e3, 1e3],
            centerAt=r'Si[111] Crystal 1',
            generator=scan,
            generatorArgs=[plots, beamline],
            startFrom=1
        )
    else:
        xrtrun.run_ray_tracing(
            beamLine=beamline,
            plots=plots,
            repeats=repeats,
            backend=r"raycing",
            generator=scan,
            generatorArgs=[plots, beamline]
        )

