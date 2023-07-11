from os import wait
from scipy.optimize import curve_fit
from typing import List
import numpy as np
import os
import matplotlib
import pickle
import functools
matplotlib.use('agg')

import xrt.backends.raycing as raycing
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.run as rrun
import xrt.runner as xrtrun
import xrt.plotter as xrtplot
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing as raycing

from components.CrocLens import CrocLens
from utils.xrtutils import get_integral_breadth
from params.sources import ring_kwargs, wiggler_nstu_scw_kwargs
from params.params_nstu_scw import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle 


mBeryllium = rm.Material('Be', rho=1.848, kind='lens')
mAl = rm.Material('Al', rho=2.7, kind='lens')
lens_material = mAl
first_lens_distance = 28000.
focus_screen_dist = 56000.
croc_L = 100.
croc_N = 100
croc_g_left = .01
croc_g_right = 1.1
num_monitors = 30


class CrocBL(raycing.BeamLine):
    def __init__(self, azimuth=0, height=0, alignE='auto'):
        super().__init__(azimuth, height, alignE)
        
        self.name = 'CrocBL' 

        self.SuperCWiggler = rsources.Wiggler(
            name=r"Superconducting Wiggler",
            bl=self,
            center=[0, 0, 0],
            eMin=100,
            eMax=100000,
            xPrimeMax=front_end_h_angle * .505e3,
            zPrimeMax=front_end_v_angle * .505e3,
            **ring_kwargs,
            **wiggler_nstu_scw_kwargs
        )

        self.FrontEnd = rapts.RectangularAperture(
            bl=self,
            name=r"Front End Slit",
            center=[0, front_end_distance, 0],
            opening=front_end_opening
        )

        self.LensStack = CrocLens.make_stack(
            L=croc_L, N=croc_N, d=croc_g_right, g_left=croc_g_left, g_right=croc_g_right,
            bl=self, 
            center=[0., first_lens_distance, 0],
            material=lens_material,
            limPhysX=[-20, 20], 
            limPhysY=[-5, 5], 
        )
        
        self.FocusMonitorStack = [
            rscreens.Screen(
                bl=self,
                name=r"Exit Monitor",
                center=[0, first_lens_distance + croc_L + ii, 0],
            ) for ii in np.logspace(2, np.log10(90000), num_monitors)

        ]

        self.ExitMonitor = rscreens.Screen(
            bl=self,
            name=r"Exit Monitor",
            center=[0, focus_screen_dist, 0],
        )


def run_process(bl: CrocBL):

    beam_source = bl.sources[0].shine()

    beam_ap1 = bl.FrontEnd.propagate(
        beam=beam_source
    )
    
    outDict = {
        'BeamSourceGlobal': beam_source,
        'BeamAperture1Local': beam_ap1,
    }

    beamIn = beam_source
    for ilens, lens in enumerate(bl.LensStack):
        lglobal, llocal1, llocal2 = lens.double_refract(beamIn, needLocal=True)
        strl = '_{0:02d}'.format(ilens)
        outDict['BeamLensGlobal'+strl] = lglobal
        outDict['BeamLensLocal1'+strl] = llocal1
        outDict['BeamLensLocal2'+strl] = llocal2
        
        llocal2a = raycing.sources.Beam(copyFrom=llocal2)
        llocal2a.absorb_intensity(beamIn)
        outDict['BeamLensLocal2a'+strl] = llocal2a
        beamIn = lglobal

    outDict['BeamMonitor'] = bl.ExitMonitor.expose(lglobal)

    for imon, mon in enumerate(bl.FocusMonitorStack):
        outDict['BeamMonitor%02d' % imon] = mon.expose(lglobal)

    bl.prepare_flow()
    return outDict

plots = [
    # Front-end
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title='FE-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title='FE-ZZpr',
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    # Focus monitor
    xrtplot.XYCPlot(
        beam='BeamMonitor',
        title='EM-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z, limits=[-.1, .1]),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor',
        title='EM-ZZpr',
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime, limits=[-2.5e-4, 2.5e-4]),
        aspect='auto'),
] + [
    xrtplot.XYCPlot(
        beam='BeamMonitor%02d' % ii,
        title='Monitor_%02d-XZ' % ii,
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto') for ii in range(num_monitors)
]  + [
    xrtplot.XYCPlot(
        beam='BeamLensLocal1_{0:02d}'.format(ii),
        title='Lens_{0:02d}-XZ'.format(ii),
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$y$', unit='mm', data=raycing.get_y),
        aspect='auto') for ii in range(croc_N)
] + [
    xrtplot.XYCPlot(
        beam='BeamLensLocal2a_{0:02d}'.format(ii),
        title='LensAbs_{0:02d}-XZ'.format(ii),
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$y$', unit='mm', data=raycing.get_y),
        aspect='auto',
        fluxKind='power') for ii in range(croc_N)
]


class AdjustFocus:
    """"""
    def __init__(self, wrapped_fn):
        self.wrapped_fn = wrapped_fn
        functools.update_wrapper(self, wrapped_fn)
    
    @staticmethod
    def fn(x, *args):
        res = np.zeros_like(x)
        res[x <= args[0]] = args[1] - (x[x <= args[0]] - args[0]) * args[2]
        res[x >= args[0]] = args[1] + (x[x >= args[0]] - args[0]) * args[3]
        return res

    def __call__(self, plts: List, bl: CrocBL):
        self.wrapped_fn = self.wrapped_fn(plts, bl)
        
        while True:
            try:
                self.wrapped_fn.__next__()
            except StopIteration:
                break

            yield
        
            # get the focus data
            dist, spot = [], []
            for plot in plts:
                if plot.persistentName is None:
                    continue

                with open(plot.persistentName, 'rb') as data:
                    data = pickle.load(data)
                    spot_ = get_integral_breadth(data)
                    if spot_ <= .2 * (data.ybinEdges[-1] - data.ybinEdges[0]):
                        spot.append(spot_)
                        dist.append(float(os.path.basename(plot.persistentName).split('-')[2]))
                os.remove(plot.persistentName)
                os.remove(plot.saveName)

            dist, spot = np.array(dist), np.array(spot)
            ii = np.argsort(dist)
            dist, spot = dist[ii], spot[ii]

            popt, _ = curve_fit(
                    self.fn, dist, spot, 
                    p0=[dist[np.argmin(spot)], np.min(spot), 1., 1.],
                    bounds=((np.min(dist), 0., -1., -1.), (np.max(dist), np.max(spot), 1., 1.)))
            print('Focal distance: %.01f mm, focus size %.01f mm' % (popt[0], popt[1]))
        
            # adjust the focus monitor position
            bl.ExitMonitor.center[1] = popt[0]
        
            # run the ray-traycing again with adjusted focus monitor
            yield

            for plot in plts:
                if plot.persistentName is not None:
                    if os.path.exists(plot.persistentName):
                        os.remove(plot.persistentName)
                    if os.path.exists(plot.saveName):
                        os.remove(plot.saveName)


# @AdjustFocus
def onept(plts: List, bl: CrocBL):
    en = 90.e3  # eV

    bl.SuperCWiggler.eMin = en - 1
    bl.SuperCWiggler.eMax = en + 1

    del bl.LensStack[:]
    fdist = first_lens_distance / 2.
    opt_pars = CrocLens.calc_optimal_params(lens_material, fdist, en)
    print(opt_pars)
    offset = 0.

    opt_L = 110.  # mm
    opt_yt = 0.7
    # y_g from 0.09 to 0.83
    print('y_g = %f' % CrocLens.calc_y_g(lens_material, fdist, en, opt_yt, opt_L))

    bl.LensStack = CrocLens.make_stack(
            L=opt_L, N=int(opt_L), d=opt_yt, g_left=offset, g_right=offset + CrocLens.calc_y_g(lens_material, fdist, en, opt_yt, opt_L),
            bl=bl, 
            center=[0., first_lens_distance, 0],
            material=lens_material,
            limPhysX=[-20, 20], 
            limPhysY=[-5, 5], 
        )

    for plot in plts:
        plot.saveName = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'croc', 'onept', plot.title + '.png')
    for plot, monitor in zip(plts[4:], bl.FocusMonitorStack):
        plot.saveName = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'croc', 'onept', plot.title + '-%.01f-' % monitor.center[1] + '.png')
        plot.persistentName = plot.saveName.replace('.png', '.pickle')
    yield


if __name__ == '__main__':
    scan = onept
    show = False
    repeats = 2

    beamline = CrocBL()
    rrun.run_process = run_process
    if show:
        beamline.glow(
            scale=[1e3, 1e4, 1e4],
            # centerAt=r'Si[111] Crystal 1',
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
