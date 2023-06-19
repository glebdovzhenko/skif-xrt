from typing import List
import os
import shutil
import numpy as np
import pickle
import matplotlib

matplotlib.use('agg')

import xrt.runner as xrtrun
import xrt.plotter as xrtplot
import xrt.backends.raycing as raycing

from utils.xrtutils import get_minmax, get_line_kb, get_integral_breadth

from NSTU_SCW import NSTU_SCW


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
    subdir = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'nstu-scw')
    scan_name = 'test'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 30.e3
    r1, r2 = -.5, -.5
    bl.MonochromatorCr1.Rx = r1 * 1e3
    bl.MonochromatorCr1.Ry = -r1 * 6e3

    bl.MonochromatorCr2.Rx = r2 * 1.e3
    bl.MonochromatorCr2.Ry = -r2 * 6.e3

    bl.align_energy(en, 5e-3)
    
    for plot in plts:
        plot.xaxis.limits = None
        plot.yaxis.limits = None
        plot.caxis.limits = None
        plot.saveName = os.path.join(subdir, scan_name, plot.title + '.png')

    yield


def r1r2_match(plts: List, bl: NSTU_SCW):
    subdir = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'nstu-scw')
    scan_name = 'r1r2_match'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 30.e3

    r1 = -1.
    for r2 in np.linspace(-.8, -1.2, 20):

        bl.MonochromatorCr1.Rx = r1 * 1e3
        bl.MonochromatorCr1.Ry = -r1 * 6e3

        bl.MonochromatorCr2.Rx = r2 * 1.e3
        bl.MonochromatorCr2.Ry = -r2 * 6.e3

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


def r1r2_scan(plts: List, bl: NSTU_SCW):
    subdir = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'nstu-scw')
    scan_name = 'r1r2_scan'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))
    
    en = 70e3

    for r in np.linspace(-1., -.9, 40):
        bl.MonochromatorCr1.Rx = r * 1e3
        bl.MonochromatorCr1.Ry = -r * 6e3

        bl.MonochromatorCr2.Rx = r * 1.e3
        bl.MonochromatorCr2.Ry = -r * 6.e3

        bl.align_energy(en, 5e-2)
    
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
        
        # fname = '%s-%dkeV-%sm-%sm.pickle' % (
        #         os.path.join(subdir, scan_name, 'FM-XXpr'), int(en * 1e-3),
        #         bl.MonochromatorCr1.pretty_R(), bl.MonochromatorCr2.pretty_R())
        # pos = bl.Cr2Monitor.center[1] 
        # dx = np.inf
        # stop = False
        # while not stop:
        #     with open(fname, 'rb') as f:
        #         data = pickle.load(f)
            
        #         k, b = get_line_kb(data, show=False)
        #         fdist = -np.sign(k) * np.sqrt((1. / k) ** 2 + (b / k) ** 2)
        #         dx_ = get_integral_breadth(data, axis='x')

        #         if dx_ > dx:
        #             bl.Cr2Monitor.center[1] = pos
        #             stop = True
        #         else:
        #             pos = bl.Cr2Monitor.center[1]
        #             dx = dx_
        #             bl.Cr2Monitor.center[1] += fdist
                
        #         # shutil.copyfile(fname.replace('.pickle', '.png'), fname.replace('.pickle', '%f.png' % fdist))
        #         for plot in plts:
        #             plot.xaxis.limits = None
        #             plot.yaxis.limits = None
        #             plot.caxis.limits = None
        #             os.remove(plot.saveName)
        #             os.remove(plot.persistentName)
        #         print(fdist)
        #         yield


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
    scan = r1r2_scan
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

