from typing import List
import os
import numpy as np
import matplotlib
# matplotlib.use('agg')

import xrt.runner as xrtrun
import xrt.plotter as xrtplot
import xrt.backends.raycing as raycing

from SKIF_1_5 import SKIF15


plots = []
x_kwds = {r'label': r'$x$', r'unit': r'mm', r'data': raycing.get_x}
z_kwds = {r'label': r'$z$', r'unit': r'mm', r'data': raycing.get_z}
xpr_kwds = {r'label': r'$x^{\prime}$', r'unit': r'', r'data': raycing.get_xprime}
zpr_kwds = {r'label': r'$z^{\prime}$', r'unit': r'', r'data': raycing.get_zprime}

for beam, t1 in zip(('BeamAperture1Local', 'BeamMonitor1Local', 'BeamMonitor2Local', 'BeamAperture2Local'), 
                    ('FE', 'C1C2', 'EM', 'ES')):
    for t2, xkw, ykw in zip(('XZ', 'XXpr', 'ZZpr'), (x_kwds, x_kwds, z_kwds), (z_kwds, xpr_kwds, zpr_kwds)):
        plots.append(xrtplot.XYCPlot(beam=beam, title='-'.join((t1, t2)), 
                                     xaxis=xrtplot.XYCAxis(**xkw), yaxis=xrtplot.XYCAxis(**ykw),
                                     aspect='auto'))

def onept(plts: List, bl: SKIF15):
    subdir = r'/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets'
    scan_name = 'de_to_e_br'
    
    r = 3.5e3
    en = 30.e3
    bl.MonochromatorCr1.R = r
    bl.MonochromatorCr2.R = r

    bl.align_energy(en, bl.get_de_over_e(r, en))
    bl.set_plot_limits(plts)

    yield
    

def e_scan(plts: List, bl: SKIF15):
    subdir = r'/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets'
    scan_name = 'e_scan'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    with open(os.path.join(subdir, scan_name + '.txt'), 'w') as f:
        tmp = bl.dumps()
        tmp += 'METADATA\n'
        tmp += 'SCAN: E\n'
        tmp += 'FILES: NAME-E-R1-R2\n'
        f.write(tmp)
    
    for en in np.arange(30., 150., 5.) * 1e3:
        bl.MonochromatorCr1.R = 57.5e3
        bl.MonochromatorCr2.R = 57.5e3

        bl.align_energy(en, bl.get_de_over_e(57.5e3, en))
        bl.set_plot_limits(plts)

        for plot in plts:
            plot.saveName = '%s-%dkeV-%sm-%sm.png' % (
                    os.path.join(subdir, scan_name, plot.title), int(en * 1e-3),
                    bl.MonochromatorCr1.pretty_R(), bl.MonochromatorCr2.pretty_R()
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')
        yield



if __name__ == '__main__':
    beamline = SKIF15()
    scan = onept
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

