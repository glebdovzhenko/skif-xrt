from typing import List
import os
import numpy as np
import pickle
import matplotlib

matplotlib.use('agg')

import xrt.runner as xrtrun
import xrt.plotter as xrtplot
import xrt.backends.raycing as raycing

from utils.xrtutils import get_minmax

from NSTU_SCW import NSTU_SCW


plots = []
x_kwds = {r'label': r'$x$', r'unit': r'mm', r'data': raycing.get_x}
z_kwds = {r'label': r'$z$', r'unit': r'mm', r'data': raycing.get_z}
xpr_kwds = {r'label': r'$x^{\prime}$', r'unit': r'', r'data': raycing.get_xprime}
zpr_kwds = {r'label': r'$z^{\prime}$', r'unit': r'', r'data': raycing.get_zprime}


for beam, t1 in zip(('BeamAperture1Local', 'BeamMonoC1Local', 'BeamMonitor1Local', 'BeamMonoC2Local', 
                     'BeamMonitor2Local'), 
                    ('FE', 'C1', 'C1C2', 'C2', 'FM')):
    for t2, xkw, ykw in zip(('XZ', 'XXpr', 'ZZpr'), (x_kwds, x_kwds, z_kwds), (z_kwds, xpr_kwds, zpr_kwds)):
        plots.append(xrtplot.XYCPlot(beam=beam, title='-'.join((t1, t2)), 
                                     xaxis=xrtplot.XYCAxis(**xkw), yaxis=xrtplot.XYCAxis(**ykw),
                                     aspect='auto'))

# for beam in sum([[b_name % (ii + 1) for ii in range(11)] for b_name in 
#                  ('BeamFilter%dLocal1', 'BeamFilter%dLocal2', 'BeamFilter%dLocal2a')], []):
    
#     t1 = beam.replace('Local1', '').replace('Local2a', '').replace('Local2', '').replace('BeamFilter', 'F')

#     if beam[-6:] == 'Local1':
#         plots.append(xrtplot.XYCPlot(beam=beam, title=(t1+'I-XZ'), aspect='auto',
#                             xaxis=xrtplot.XYCAxis(label='$x$', unit='mm', data=raycing.get_x),
#                             yaxis=xrtplot.XYCAxis(label='$y$', unit='mm', data=raycing.get_y)))
#         plots.append(xrtplot.XYCPlot(beam=beam, title=(t1+'P-XZ'), aspect='auto', fluxKind='power',
#                             xaxis=xrtplot.XYCAxis(label='$x$', unit='mm', data=raycing.get_x),
#                             yaxis=xrtplot.XYCAxis(label='$y$', unit='mm', data=raycing.get_y)))
#     elif beam[-6:] == 'Local2':
#         plots.append(xrtplot.XYCPlot(beam=beam, title=(t1+'IT-XZ'), aspect='auto',
#                             xaxis=xrtplot.XYCAxis(label='$x$', unit='mm', data=raycing.get_x),
#                             yaxis=xrtplot.XYCAxis(label='$y$', unit='mm', data=raycing.get_y)))
#     elif beam[-7:] == 'Local2a':
#         plots.append(xrtplot.XYCPlot(beam=beam, title=(t1+'PA-XZ'), aspect='auto', fluxKind='power',
#                             xaxis=xrtplot.XYCAxis(label='$x$', unit='mm', data=raycing.get_x),
#                             yaxis=xrtplot.XYCAxis(label='$y$', unit='mm', data=raycing.get_y)))
#     else:
#         pass
# else:
#     plots.append(xrtplot.XYCPlot(beam='BeamMonoC1Local2a', title=('C1PA-XZ'), aspect='auto', fluxKind='power',
#                  xaxis=xrtplot.XYCAxis(label='$x$', unit='mm', data=raycing.get_x),
#                  yaxis=xrtplot.XYCAxis(label='$y$', unit='mm', data=raycing.get_y)))


def onept(plts: List, bl: NSTU_SCW):
    subdir = r'/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/nstu-scw'
    scan_name = 'test'
    
    en = 30.e3
    bl.MonochromatorCr1.R = 10.e3
    bl.MonochromatorCr2.R = 20.e3

    bl.align_energy(en, 5e-3)
    # bl.set_plot_limits(plts)

    yield


def get_focus(plts: List, bl: NSTU_SCW):
    subdir = r'/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/nstu-scw'
    scan_name = 'double_-14R1'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))
    if not os.path.exists(os.path.join(subdir, scan_name + '_')):
        os.mkdir(os.path.join(subdir, scan_name + '_'))
    en = 30.e3
    # cr1_rs = [25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 90.]
    # r2 = -1.5
    # for r1 in cr1_rs:
    # r1 = 35.
    # cr2_rs = [-.6, -.7, -.8, -.9, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.1, -2.2, -2.3, -2.4, -2.5, -2.6, -2.7, -2.8, -2.9]
    # for r2 in cr2_rs:
    for r1 in [-1.4]:
        for r2 in np.linspace(-.5, -2., 15):
            bl.MonochromatorCr2.Rx = 1e3 * r2
            bl.MonochromatorCr2.Ry = -6e3 * r2
            bl.MonochromatorCr1.Ry = 1e3 * r1
            bl.align_energy(en, 1e-1)

            for plot in plts:
                el, crd = plot.title.split('-')
                if (el not in ('FM', 'EM', 'C1C2')) or (crd not in ('XXpr', 'ZZpr', 'XZ')):
                    continue
            
                plot.xaxis.limits = None
                plot.yaxis.limits = None
                plot.caxis.limits = None
                plot.saveName = os.path.join(subdir, scan_name, 
                                        plot.title  + '-%sm' % bl.MonochromatorCr1.pretty_R() + \
                                                '-%sm' % bl.MonochromatorCr2.pretty_R() + '.png'
                                        )
                plot.persistentName = plot.saveName.replace('.png', '.pickle')
            if os.path.exists(plot.saveName):
                continue
        
            yield
        
            bl.align_energy(en, 1e-10)
            ax_mul = 1.1

            for plot in plts:
                el, crd = plot.title.split('-')
                if plot.saveName:
                    data = pickle.load(open(plot.persistentName, 'rb'))
                
                    x1, x2 = get_minmax(data, axis='x')
                    y1, y2 = get_minmax(data, axis='y')
                    e1, e2 = get_minmax(data, axis='e')

                    x1, x2 = x2 - ax_mul * np.abs(x1 - x2), x1 + ax_mul * np.abs(x1 - x2)
                    y1, y2 = y2 - ax_mul * np.abs(y1 - y2), y1 + ax_mul * np.abs(y1 - y2)
                    e1, e2 = e2 - ax_mul * np.abs(e1 - e2), e1 + ax_mul * np.abs(e1 - e2)

                    plot.xaxis.limits = [x1, x2]
                    plot.yaxis.limits = [y1, y2]
                    plot.caxis.limits = [e1, e2]

                    if bl.SuperCWiggler.eMin > e1:
                        bl.SuperCWiggler.eMin = e1
                    if bl.SuperCWiggler.eMax < e2:
                        bl.SuperCWiggler.eMax = e2

                    plot.persistentName = plot.persistentName.replace(scan_name, scan_name + '_')
                    plot.saveName = plot.saveName.replace(scan_name, scan_name + '_')

            yield


if __name__ == '__main__':
    beamline = NSTU_SCW()
    scan = get_focus
    show = False
    repeats = 10

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

