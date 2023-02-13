import numpy as np
from copy import deepcopy

import matplotlib

import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.myopencl as mcl
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from components import BentLaueCylinder, CrystalSiPrecalc
from params.sources import ring_kwargs, wiggler_nstu_scw_kwargs
from params.params_NSTU_SCW import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle, \
    monochromator_distance, monochromator_z_offset, monochromator_x_lim, monochromator_y_lim, crl_distance


# ################################################# SETUP PARAMETERS ###################################################


""" Monochromator """
monochromator_alignment_energy = 30.e3
monochromator_c1_alpha = np.radians(35.3)
monochromator_c1_thickness = .2
monochromator_c2_alpha = np.radians(35.3)
monochromator_c2_thickness = 2.0


# #################################################### MATERIALS #######################################################


cr_si_1 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c1_thickness, 
                           database='/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/components/Si111ref_sag.csv',
                           mirrorRs=True)
# cr_si_2 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c2_thickness, 
#                        database='/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/components/Si111ref_sag.csv')
cr_si_2 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c2_thickness)


# #################################################### BEAMLINE ########################################################


class NSTU_SCW(raycing.BeamLine):
    def __init__(self):
        raycing.BeamLine.__init__(self)

        self.name = "NSTU SCW"

        self.alignE = monochromator_alignment_energy

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

        self.MonochromatorCr1 = BentLaueCylinder(
            bl=self,
            name=r'Si[111] Crystal 1',
            center=[0., monochromator_distance, 0.],
            pitch=np.pi / 2.,
            roll=0.,
            yaw=0.,
            alpha=monochromator_c1_alpha,
            material=(cr_si_1,),
            bendingOrientation='sagittal',
            R=np.inf,
            targetOpenCL='CPU',
            limPhysY=monochromator_y_lim,
            limOptY=monochromator_y_lim,
            limPhysX=monochromator_x_lim,
            limOptX=monochromator_x_lim,
        )
        self.MonochromatorCr1.ucl = mcl.XRT_CL(r'materials.cl', targetOpenCL='CPU')

        self.Cr1Monitor = rscreens.Screen(
            bl=self,
            name=r"Crystal 1 Monitor",
            center=[0, monochromator_distance, .5 * monochromator_z_offset],
        )

        self.MonochromatorCr2 = BentLaueCylinder(
            bl=self,
            name=r'Si[111] Crystal 2',
            center=[0., monochromator_distance, monochromator_z_offset],
            positionRoll=np.pi,
            pitch=0.,
            roll=0.,
            yaw=0.,
            alpha=monochromator_c2_alpha,
            material=(cr_si_2,),
            bendingOrientation='meridional',
            R=np.inf,
            targetOpenCL='CPU',
            limPhysY=monochromator_y_lim,
            limOptY=monochromator_y_lim,
            limPhysX=monochromator_x_lim,
            limOptX=monochromator_x_lim,
        )
        self.MonochromatorCr2.ucl = mcl.XRT_CL(r'materials.cl', targetOpenCL='CPU')

        self.Cr2Monitor = rscreens.Screen(
            bl=self,
            name=r"Crystal 2 Monitor",
            center=[0, crl_distance - 10, monochromator_z_offset],
        )

    def print_positions(self):
        print('#' * 20, self.name, '#' * 20)

        for element in (self.SuperCWiggler, self.FrontEnd,
                        self.MonochromatorCr1, self.Cr1Monitor, self.MonochromatorCr2):
            print('#' * 5, element.name, 'at', element.center)

        for element in (self.MonochromatorCr1, self.MonochromatorCr2):
            print('#' * 5, element.name, 'RxRyRz', element.pitch, element.roll, element.yaw)

        print('#' * (42 + len(self.name)))


    def align_energy(self, en, d_en, mono=False):
        # changing energy for the beamline / source
        self.alignE = en
        if not mono:
            self.SuperCWiggler.eMin = en * (1. - d_en)
            self.SuperCWiggler.eMax = en * (1. + d_en)
        else:
            self.SuperCWiggler.eMin = en - 1.
            self.SuperCWiggler.eMax = en + 1.

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

        # between-crystals monitor
        self.Cr1Monitor.center = [
            0.,
            monochromator_distance + .5 * monochromator_z_offset / np.tan(2. * theta0),
            .5 * monochromator_z_offset
        ]

        self.print_positions()




# ################################################# BEAM TOPOLOGY ######################################################


def run_process(bl: NSTU_SCW):

    beam_source = bl.sources[0].shine()

    beam_ap1 = bl.FrontEnd.propagate(
        beam=beam_source
    )

    beam_mono_c1_global, beam_mono_c1_local = bl.MonochromatorCr1.reflect(
        beam=beam_source
    )

    beam_mon1 = bl.Cr1Monitor.expose(
        beam=beam_mono_c1_global
    )

    beam_mono_c2_global, beam_mono_c2_local = bl.MonochromatorCr2.reflect(
        beam=beam_mono_c1_global
    )

    beam_mon2 = bl.Cr2Monitor.expose(
        beam=beam_mono_c2_global
    )

    bl.prepare_flow()

    return {
        'BeamSourceGlobal': beam_source,
        'BeamAperture1Local': beam_ap1,
        'BeamMonoC1Local': beam_mono_c1_local,
        'BeamMonoC1Global': beam_mono_c1_global,
        'BeamMonitor1Local': beam_mon1,
        'BeamMonoC2Local': beam_mono_c2_local,
        'BeamMonoC2Global': beam_mono_c2_global,
        'BeamMonitor2Local': beam_mon2,
    }


rrun.run_process = run_process

