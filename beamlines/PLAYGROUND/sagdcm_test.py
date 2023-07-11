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

from components import BentLaueParaboloid, CrystalSiPrecalc
from utils.xrtutils import get_integral_breadth
from params.params_nstu_scw import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle, \
    monochromator_distance, monochromator_z_offset, monochromator_x_lim, monochromator_y_lim, \
    croc_crl_distance, croc_crl_L, croc_crl_y_t, exit_slit_distance


num_monitors = 30
# ################################################# SETUP PARAMETERS ###################################################


""" Monochromator """
monochromator_alignment_energy = 30.e3
monochromator_c1_alpha = np.radians(35.3)
monochromator_c1_thickness = .5
monochromator_c2_alpha = np.radians(35.3)
monochromator_c2_thickness = .5


# #################################################### MATERIALS #######################################################


cr_si_1 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c1_thickness,
                           database=os.path.join(os.getenv('BASE_DIR'), 'components', 'Si111ref_sag.csv'))
cr_si_2 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c2_thickness,
                           database=os.path.join(os.getenv('BASE_DIR'), 'components', 'Si111ref_sag.csv'))


class SagDCMTestBL(raycing.BeamLine):
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

        self.MonochromatorCr1 = BentLaueParaboloid(
            bl=self,
            name=r'Si[111] Crystal 1',
            center=[0., monochromator_distance, 0.],
            pitch=np.pi / 2.,
            roll=0.,
            yaw=0.,
            alpha=monochromator_c1_alpha,
            material=(cr_si_1,),
            r_for_refl='x',
            targetOpenCL='CPU',
            limPhysY=monochromator_y_lim,
            limOptY=monochromator_y_lim,
            limPhysX=monochromator_x_lim,
            limOptX=monochromator_x_lim,
        )

        self.MonochromatorCr2 = BentLaueParaboloid(
            bl=self,
            name=r'Si[111] Crystal 2',
            center=[0., monochromator_distance, monochromator_z_offset],
            positionRoll=np.pi,
            pitch=0.,
            roll=0.,
            yaw=0.,
            alpha=monochromator_c2_alpha,
            material=(cr_si_2,),
            r_for_refl='x',
            targetOpenCL='CPU',
            limPhysY=monochromator_y_lim,
            limOptY=monochromator_y_lim,
            limPhysX=monochromator_x_lim,
            limOptX=monochromator_x_lim,
        )

        self.FocusMonitorStack = [
            rscreens.Screen(
                bl=self,
                name=r"Exit Monitor",
                center=[0, monochromator_distance + 2000 + ii, monochromator_z_offset],
            ) for ii in np.logspace(2, np.log10(40000), num_monitors)
        ]


    def align_energy(self, en):
        # Diffraction angle for the DCM
        theta0 = np.arcsin(rm.ch / (2 * self.MonochromatorCr1.material[0].d * en))

        # Setting up DCM orientations / positions
        # Crystal 1
        self.MonochromatorCr1.pitch = np.pi / 2 + theta0 + self.MonochromatorCr1.alpha
        self.MonochromatorCr1.center = [
            0.,
            monochromator_distance,
            0.
        ]

        # Crystal 2
        self.MonochromatorCr2.pitch = np.pi / 2 - theta0 + self.MonochromatorCr2.alpha
        self.MonochromatorCr2.center = [
            0.,
            monochromator_distance + monochromator_z_offset / np.tan(2. * theta0),
            monochromator_z_offset
        ]

        for mon, ii in zip(self.FocusMonitorStack, np.logspace(2, np.log10(30000), num_monitors)):
            mon.center = [0, monochromator_distance + monochromator_z_offset / np.tan(2. * theta0) + ii, monochromator_z_offset]


def run_process(bl: SagDCMTestBL):
    outDict = {
        'BeamSourceGlobal': bl.sources[0].shine(),
    }

    outDict['BeamSourceScreenLocal'] = bl.SourceScreen.expose(
        outDict['BeamSourceGlobal']
    )
    
    beamIn = outDict['BeamSourceGlobal']
    beam_mono_c1_global, beam_mono_c1_local = bl.MonochromatorCr1.reflect(
        beam=beamIn
    )

    beam_mono_c2_global, beam_mono_c2_local = bl.MonochromatorCr2.reflect(
        beam=beam_mono_c1_global
    )

    outDict['BeamMonoC1Local'] = beam_mono_c1_local
    outDict['BeamMonoC1Global'] = beam_mono_c1_global
    outDict['BeamMonoC2Local'] = beam_mono_c2_local
    outDict['BeamMonoC2Global'] = beam_mono_c2_global

    for imon, mon in enumerate(bl.FocusMonitorStack):
        outDict['BeamMonitor%02d' % imon] = mon.expose(beam_mono_c2_global)

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
] + [
    xrtplot.XYCPlot(
        beam='BeamMonitor%02d' % ii,
        title='Monitor_%02d-XZ' % ii,
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto') for ii in range(num_monitors)
] 


def onept(plts: List, bl: SagDCMTestBL):
    bl.GSource.energies = (30.e3, )
    r1, r2 = -1., -1.
    
    bl.MonochromatorCr1.Rx = r1 * 1e3
    bl.MonochromatorCr1.Ry = -r1 * 6e3
    bl.MonochromatorCr2.Rx = r2 * 1.e3
    bl.MonochromatorCr2.Ry = -r2 * 6.e3
    bl.align_energy(bl.GSource.energies[0])

    for plot in plts:
        plot.saveName = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'sagdcm_test', plot.title + '.png')
        # plot.persistentName = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'croc_test', plot.title + '.pickle')
    for plot, monitor in zip(plts[3:], bl.FocusMonitorStack):
        plot.saveName = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'sagdcm_test', plot.title + '-%.01f-' % monitor.center[1] + '.png')
        plot.persistentName = plot.saveName.replace('.png', '.pickle')
    yield


if __name__ == '__main__':
    scan = onept
    show = False
    repeats = 2

    beamline = SagDCMTestBL()
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
