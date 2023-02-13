from typing import List
import os
import numpy as np
import pickle
import matplotlib

# matplotlib.use('agg')

import xrt.runner as xrtrun
import xrt.plotter as xrtplot
import xrt.backends.raycing as raycing

from utils.xrtutils import get_line_kb

from NSTU_SCW import NSTU_SCW


plots = []
x_kwds = {r'label': r'$x$', r'unit': r'mm', r'data': raycing.get_x}
z_kwds = {r'label': r'$z$', r'unit': r'mm', r'data': raycing.get_z}
xpr_kwds = {r'label': r'$x^{\prime}$', r'unit': r'', r'data': raycing.get_xprime}
zpr_kwds = {r'label': r'$z^{\prime}$', r'unit': r'', r'data': raycing.get_zprime}


for beam, t1 in zip(('BeamAperture1Local', 'BeamMonoC1Local', 'BeamMonitor1Local', 'BeamMonoC2Local', 
                     'BeamMonitor2Local'), 
                    ('FE', 'C1', 'C1C2', 'C2', 'EM')):
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
    subdir = r'/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets'
    scan_name = 'test'
    
    en = 30.e3
    bl.MonochromatorCr1.R = 10.e3
    bl.MonochromatorCr2.R = 20.e3

    bl.align_energy(en, 5e-3)
    # bl.set_plot_limits(plts)

    yield


def get_focus(plts: List, bl: NSTU_SCW):
    subdir = r'/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/nstu-scw'
    scan_name = 'get_focus'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))
    else:
        for f_name in os.listdir(os.path.join(subdir, scan_name)):
            os.remove(os.path.join(subdir, scan_name, f_name))
    
    en = 30.e3
    bl.MonochromatorCr1.R = -2.e3
    bl.MonochromatorCr2.R = np.inf
    bl.align_energy(en, 5e-3)

    for plot in plts:
        el, crd = plot.title.split('-')
        if (el not in ('FE', 'EM', 'C1C2')) or (crd not in ('XXpr', 'ZZpr')):
            continue

        plot.saveName = os.path.join(subdir, scan_name, plot.title + '.png')
        plot.persistentName = plot.saveName.replace('.png', '.pickle')
    else:
        yield

    for f_name in sorted(filter(lambda x: x[-7:] == '.pickle', os.listdir(os.path.join(subdir, scan_name)))):
        with open(os.path.join(subdir, scan_name, f_name), 'rb') as f:
            data = pickle.load(f)
            k, b = get_line_kb(data)
            dist = -np.sign(k) * np.sqrt((1. / k) ** 2 + (b / k) ** 2)

            if f_name.split('-')[0] == 'C1C2':
                dist -= np.sqrt(np.sum((np.array(bl.Cr1Monitor.center) - np.array(bl.MonochromatorCr1.center))**2))
            elif f_name.split('-')[0] == 'EM':
                dist -= np.sqrt(np.sum((np.array(bl.Cr2Monitor.center) - np.array(bl.MonochromatorCr2.center))**2))

            print(f_name, dist)


def focus_scan(plts: List, bl: NSTU_SCW):
    subdir = r'/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/nstu-scw'
    scan_name = 'focus_scan'

    yield
    

if __name__ == '__main__':
    beamline = NSTU_SCW()
    scan = get_focus
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

