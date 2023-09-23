from typing import List
import os
import shutil
import numpy as np
import matplotlib
import pickle

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


def locate_focus(plts: List, bl: NSTU_SCW):
    data_dir = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'tmp')

    def slice_parabola(a, b, c, m):
        m += 1.
        x0 = -b / (2. * c)
        a_ = a + m * (b * b / (4 * c) - a)
        d = np.sqrt(b * b - 4 * a_ * c)
        x1 = (-b - d) / (2. * c)
        x2 = (-b + d) / (2. * c)
        return x0, x1, x2
    
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

    bl.reset_screen_stack()
    for ii in range(len(plts) - 1, -1, -1):
        if 'BeamFSSLocal' in plts[ii].title:
            del plts[ii]

    plts.extend([
        xrtplot.XYCPlot(
            beam='BeamFSSLocal_{0:02d}'.format(iscreen), 
            title='BeamFSSLocal_{0:02d}'.format(iscreen), 
            persistentName=os.path.join(data_dir, 'BeamFSSLocal_%.03f.pickle' % screen.center[1]),
            saveName=os.path.join(data_dir, 'BeamFSSLocal_%.03f.png' % screen.center[1]),
            aspect='auto',
            xaxis=xrtplot.XYCAxis(label='$x$', unit='mm', data=raycing.get_x), 
            yaxis=xrtplot.XYCAxis(label='$z$', unit='mm', data=raycing.get_z, limits=[-.5, .5])) 
        for iscreen, screen in enumerate(bl.FScreenStack)
    ])

    onept(plts, bl).__next__()
    yield

    # calculating focus position and size
    pos, y_size, x_size = [], [], []
    for f_name in (os.path.join(data_dir, 'BeamFSSLocal_%.03f.pickle' % screen.center[1]) 
                    for screen in bl.FScreenStack):
        with open(f_name, 'rb') as f:
            f = pickle.load(f)
            y_size.append(get_integral_breadth(f, 'y'))
            x_size.append(get_integral_breadth(f, 'x'))
            pos.append(float(os.path.basename(f_name).replace('.pickle', '').replace('BeamFSSLocal_', '')))
    else:
        pos, y_size, x_size = np.array(pos), np.array(y_size), np.array(x_size)
        ii = np.argsort(pos)
        pos, y_size, x_size = pos[ii], y_size[ii], x_size[ii]

    pp_y = np.polynomial.polynomial.Polynomial.fit(pos, y_size, 2)
    pp_x = np.polynomial.polynomial.Polynomial.fit(pos, x_size, 2)

    coef_y = pp_y.convert().coef
    coef_x = pp_x.convert().coef

    focus_y, _, _ = slice_parabola(*coef_y, 0.1)
    focus_x, _, _ = slice_parabola(*coef_x, 0.1)

    fig, (ax1, ax2) = matplotlib.pyplot.subplots(2, 1)
    ax2.plot(pos, y_size)
    ax1.plot(pos, x_size)
    ax1.plot(pos, pp_x(pos))
    ax2.plot(pos, pp_y(pos))

    ax2.plot([focus_y, focus_y], [y_size.min(), y_size.max()], '--')
    ax2.text(focus_y, y_size.max(), 'Fz=%.01f mm' % focus_y)
    ax1.plot([focus_x, focus_x], [x_size.min(), x_size.max()], '--')
    ax1.text(focus_x, x_size.max(), 'Fx=%.01f mm' % focus_x)

    ax2.set_xlabel('Y position [mm]')
    ax2.set_ylabel('Beam integral breadth [mm]')
    ax1.set_xlabel('Y position [mm]')
    ax1.set_ylabel('Beam integral breadth [mm]')

    fig.savefig(os.path.join(data_dir, '..', 'fdist.png'))



def onept(plts: List, bl: NSTU_SCW):
    subdir = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'nstu-scw')
    scan_name = 'spot_size'

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

    bl.align_energy(en, 3e-2, invert_croc=False)  # 5e-3 30 50 kev; 1e-2 70 kev; 3e-2 90keV
    # bl.SuperCWiggler.eMin = 100.
    # bl.SuperCWiggler.eMax = 1.e6

    del bl.CrocLensStack[:]
    # g_f = 1.076  # 1.228  # 30 keV
    # g_f = 0.39 #.435  # 50 keV
    # g_f = .191  # .224  # 70 keV
    g_f = .101  # .138  # 90 keV
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
            plot.yaxis.limits = [-.2, .2]
            plot.caxis.limits = [bl.SuperCWiggler.eMin, bl.SuperCWiggler.eMax]
            plot.saveName = os.path.join(subdir, scan_name, '%s-%dkeV.png' % (plot.title, int(en * 1e-3)))
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

    yield


if __name__ == '__main__':
    beamline = NSTU_SCW()
    scan = onept
    show = False
    repeats = 38

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

