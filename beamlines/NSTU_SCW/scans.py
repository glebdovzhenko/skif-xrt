from typing import List
import os
import numpy as np
import matplotlib

matplotlib.use('agg')

import xrt.runner as xrtrun
import xrt.plotter as xrtplot
import xrt.backends.raycing as raycing

from utils.xrtutils import get_minmax, get_line_kb, get_integral_breadth

from NSTU_SCW import NSTU_SCW
from params.params_nstu_scw import croc_crl_L, sic_filter_N, diamond_filter_N


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

# # Adding crl plots
# plots.extend([xrtplot.XYCPlot(
#         beam='BeamLensLocal2a_{0:02d}'.format(ii),
#         title='LensAbs_{0:02d}-XZ'.format(ii),
#         xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
#         yaxis=xrtplot.XYCAxis(label=r'$y$', unit='mm', data=raycing.get_y),
#         aspect='auto',
#         fluxKind='power') for ii in range(int(croc_crl_L))])
# # Adding filter plots
# plots.extend([xrtplot.XYCPlot(
#         beam='BeamFilterCLocal2a_{0:02d}'.format(ii),
#         title='FilterCAbs_{0:02d}-XZ'.format(ii),
#         xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
#         yaxis=xrtplot.XYCAxis(label=r'$y$', unit='mm', data=raycing.get_y),
#         aspect='auto',
#         fluxKind='power') for ii in range(diamond_filter_N)])
# plots.extend([xrtplot.XYCPlot(
#         beam='BeamFilterSiCLocal2a_{0:02d}'.format(ii),
#         title='FilterSiCAbs_{0:02d}-XZ'.format(ii),
#         xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
#         yaxis=xrtplot.XYCAxis(label=r'$y$', unit='mm', data=raycing.get_y),
#         aspect='auto',
#         fluxKind='power') for ii in range(sic_filter_N)])


def onept(plts: List, bl: NSTU_SCW):
    subdir = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'nstu-scw')
    scan_name = 'lens_abs_90'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 90.e3
    # r1, r2 = -2.04, -2.04  # 30 keV
    # r1, r2 = -1.22, -1.22  # 50 keV
    # r1, r2 = -.87, -.87  # 70 keV
    r1, r2 = -.675, -.675  # 90 keV
    bl.MonochromatorCr1.Rx = r1 * 1e3
    bl.MonochromatorCr1.Ry = -r1 * 6e3

    bl.MonochromatorCr2.Rx = r2 * 1.e3
    bl.MonochromatorCr2.Ry = -r2 * 6.e3

    bl.align_energy(en, 5e-2, invert_croc=False)
    bl.SuperCWiggler.eMin = 100.
    bl.SuperCWiggler.eMax = 1.e6
    
    for plot in plts:
        if plot.title == 'FM-XZ':
            plot.xaxis.limits = [-.5, .5]
            plot.yaxis.limits = [-.5, .5]
        # plot.caxis.limits = None
        plot.saveName = os.path.join(subdir, scan_name, plot.title + '.png')
        plot.persistentName = plot.saveName.replace('.png', '.pickle')

    yield


def r1r2_scan(plts: List, bl: NSTU_SCW):
    subdir = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'nstu-scw')
    scan_name = 'r1r2_scan'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))
    
    en = 90e3

    for r in np.linspace(-.668, -.689, 20):
        bl.MonochromatorCr1.Rx = r * 1e3
        bl.MonochromatorCr1.Ry = -r * 6e3

        bl.MonochromatorCr2.Rx = r * 1.e3
        bl.MonochromatorCr2.Ry = -r * 6.e3

        bl.align_energy(en, 2e-2)
    
        for plot in plts:
            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None
            plot.saveName = '%s-%dkeV-%sm-%sm.png' % (
                os.path.join(subdir, scan_name, plot.title), int(en * 1e-3),
                bl.MonochromatorCr1.pretty_R(), bl.MonochromatorCr2.pretty_R()
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        yield


def e_scan(plts: List, bl: NSTU_SCW):
    subdir = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'nstu-scw')
    scan_name = 'mask_lens_e_scan_gr'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))     

    for en, d_en, r in zip([30., 50., 70., 90.], [2e-3, 4e-3, 7e-3, 2e-2], [-2.04, -1.22, -.87, -.675]):
        bl.MonochromatorCr1.Rx = r * 1e3
        bl.MonochromatorCr1.Ry = -r * 6e3
        bl.MonochromatorCr2.Rx = r * 1.e3
        bl.MonochromatorCr2.Ry = -r * 6.e3
        
        bl.align_energy(en * 1e3, d_en, invert_croc=True)
        # bl.SuperCWiggler.eMin = 100.
        # bl.SuperCWiggler.eMax = 1.e6
        
        for plot in plts:
            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None

            if plot.title == 'FM-XZ':
                plot.xaxis.limits = [-.5, .5]
                plot.yaxis.limits = [-.5, .5]

            plot.saveName = '%s-%dkeV-%sm-%sm.png' % (
                os.path.join(subdir, scan_name, plot.title), int(en),
                bl.MonochromatorCr1.pretty_R(), bl.MonochromatorCr2.pretty_R()
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        yield


if __name__ == '__main__':
    beamline = NSTU_SCW()
    scan = e_scan
    show = False
    repeats = 1

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

