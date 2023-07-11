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
first_lens_distance = 24000.
focus_screen_dist = 30000.
croc_L = 100.
croc_N = 100
croc_d = 1.1
croc_g_left = 0.
croc_g_right = 1.1
num_monitors = 30


class CrocTestBL(raycing.BeamLine):
    def __init__(self, azimuth=0, height=0, alignE='auto'):
        super().__init__(azimuth, height, alignE)
        
        self.name = 'CrocTestBL' 

        self.GSource = rsources.GeometricSource(
            bl=self,
            name='Source',
            center=[0, 0, 0],
            dx=[-10,10],
            dy=0,
            dz=[-2.5, 2.5],
            dxprime=0,
            dzprime=0,
            distE='lines',
            distx='flat', 
            # disty='flat', 
            distz='flat', 
            # distxprime=None, 
            # distzprime=None
        )

        self.SourceScreen = rscreens.Screen(
            name='SourceScreen',
            bl=self,
            center=[0, 1000, 0]
        )

        self.LensStack = CrocLens.make_stack(
            L=croc_L, N=croc_N, d=croc_d, g_left=croc_g_left, g_right=croc_g_right,
            bl=self, 
            center=[0., 2000, 0],
            material=mBeryllium,
            limPhysX=[-20, 20], 
            limPhysY=[-5, 5], 
        )

        self.FocusMonitorStack = [
            rscreens.Screen(
                bl=self,
                name=r"Exit Monitor",
                center=[0, 2000 + croc_L + ii, 0],
            ) for ii in np.logspace(2, np.log10(400000), num_monitors)

        ]

        self.ExitMonitor = rscreens.Screen(
            bl=self,
            name=r"Exit Monitor",
            center=[0, 5000, 0],
        )

        self.LensMonitor = rscreens.Screen(
            bl=self,
            name=r"Lens Monitor",
            center=[0, 2000 + croc_L + 2 * croc_L / croc_N, 0],
        )


def run_process(bl: CrocTestBL):

    outDict = {
        'BeamSourceGlobal': bl.sources[0].shine(),
    }

    outDict['BeamSourceScreenLocal'] = bl.SourceScreen.expose(
        outDict['BeamSourceGlobal']
    )
    
    beamIn = outDict['BeamSourceGlobal']
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
    
    outDict['BeamFocusScreenLocal'] = bl.ExitMonitor.expose(lglobal)
    outDict['BeamLensScreenLocal'] = bl.LensMonitor.expose(lglobal)

    for imon, mon in enumerate(bl.FocusMonitorStack):
        outDict['BeamMonitor%02d' % imon] = mon.expose(lglobal)

    bl.prepare_flow()
    return outDict


plots = [
    # Source
    xrtplot.XYCPlot(
        beam='BeamSourceScreenLocal',
        title='FE-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamSourceScreenLocal',
        title='FE-ZZpr',
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamSourceScreenLocal',
        title='FE-XXpr',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        aspect='auto'),
    # Focus monitor
    xrtplot.XYCPlot(
        beam='BeamFocusScreenLocal',
        title='EM-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamFocusScreenLocal',
        title='EM-ZZpr',
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    # sigma_rms monitor
    xrtplot.XYCPlot(
        beam='BeamLensScreenLocal',
        title='LE-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
] + [
    xrtplot.XYCPlot(
        beam='BeamMonitor%02d' % ii,
        title='Monitor_%02d-XZ' % ii,
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto') for ii in range(num_monitors)
] 


def onept(plts: List, bl: CrocTestBL):
    bl.GSource.energies = (90.e3, )

    focus_dist = croc_d * croc_g_right / (croc_L * np.real(1. - mBeryllium.get_refractive_index(bl.GSource.energies[0])))
    sigma_rms = np.sqrt(focus_dist * 
                        np.real(1. - mBeryllium.get_refractive_index(bl.GSource.energies[0])) * 
                        10 / mBeryllium.get_absorption_coefficient(bl.GSource.energies[0]))
    bl.ExitMonitor.center[1] = 2000 + croc_L / 2 + focus_dist
    for mon, pos in zip(
        bl.FocusMonitorStack, 
        np.linspace(
            bl.ExitMonitor.center[1] - .25 * focus_dist, 
            bl.ExitMonitor.center[1] + .25 * focus_dist, 
            len(bl.FocusMonitorStack)
        )):
        mon.center[1] = pos

    print('Focus: %.01f, Monitor: %.01f, exit FWHM %.01f' % (focus_dist, bl.ExitMonitor.center[1], 2.355 * sigma_rms))
    for plot in plts:
        plot.saveName = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'croc_test', plot.title + '.png')
        # plot.persistentName = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'croc_test', plot.title + '.pickle')
    for plot, monitor in zip(plts[6:], bl.FocusMonitorStack):
        plot.saveName = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'croc_test', plot.title + '-%.01f-' % monitor.center[1] + '.png')
        plot.persistentName = plot.saveName.replace('.png', '.pickle')
    yield


if __name__ == '__main__':
    scan = onept
    show = False
    repeats = 2

    beamline = CrocTestBL()
    rrun.run_process = run_process
    if show:
        beamline.glow(
            scale=[1e3, 1e4, 1e4],
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
