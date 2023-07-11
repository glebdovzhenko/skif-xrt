import numpy as np
from copy import deepcopy
import os

import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.backends.raycing.oes as roe

from components import CrystalSiPrecalc, BentLaueParaboloid, CrocLens
from params.sources import ring_kwargs, wiggler_nstu_scw_kwargs, wiggler_1_5_kwargs
from params.params_nstu_scw import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle, \
    filter_distance, diamond_filter_th, diamond_filter_N, sic_filter_th, sic_filter_N, \
    monochromator_distance, monochromator_z_offset, monochromator_x_lim, monochromator_y_lim, \
    croc_crl_distance, croc_crl_L, croc_crl_y_t, exit_slit_distance, crl_mask_distance


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
mBeryllium = rm.Material('Be', rho=1.848, kind='lens')
mAl = rm.Material('Al', rho=2.7, kind='lens')
mDiamond = rm.Material('C', rho=3.5, kind='lens')
mGraphite = rm.Material('C', rho=2.15, kind='lens')
mDiamondF = rm.Material('C', rho=3.5)
mSiC = rm.Material(('Si', 'C'), quantities=(1, 1), rho=3.16)
lens_material = mGraphite


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
            # **wiggler_nstu_scw_kwargs
            **wiggler_1_5_kwargs
        )

        self.FrontEnd = rapts.RectangularAperture(
            bl=self,
            name=r"Front End Slit",
            center=[0, front_end_distance, 0],
            opening=front_end_opening
        )

        self.FilterStackC = []
        self.FilterStackSiC = []

        for ii in range(diamond_filter_N):
            self.FilterStackC.append(
                roe.Plate(
                    name='Diamond Filter %d' % (ii + 1),
                    bl=self,
                    center=[0, filter_distance + ii * 1.1 * diamond_filter_th, 0],
                    pitch=np.pi/2.,
                    material=mDiamondF,
                    t=diamond_filter_th
                )
            )

        for ii in range(sic_filter_N):
            self.FilterStackSiC.append(
                roe.Plate(
                    name='SiC Filter %d' % (ii + 1),
                    bl=self,
                    center=[0, filter_distance + diamond_filter_N * 1.1 * diamond_filter_th + ii * 1.1 * sic_filter_th, 0],
                    pitch=np.pi/2.,
                    material=mSiC,
                    t=sic_filter_th
                )
            )

        self.CrlMask = rapts.RectangularAperture(
            bl=self,
            name=r"Front End Slit",
            center=[0, crl_mask_distance, 0],
            opening=front_end_opening
        )

        self.CrocLensStack = CrocLens.make_stack(
            L=croc_crl_L, N=int(croc_crl_L), d=croc_crl_y_t, g_left=0., g_right=croc_crl_y_t,
            bl=self, 
            center=[0., croc_crl_distance, 0],
            material=lens_material,
            limPhysX=monochromator_x_lim, 
            limPhysY=monochromator_y_lim, 
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

        self.Cr1Monitor = rscreens.Screen(
            bl=self,
            name=r"Crystal 1 Monitor",
            center=[0, monochromator_distance, .5 * monochromator_z_offset],
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

        self.Cr2Monitor = rscreens.Screen(
            bl=self,
            name=r"Crystal 2 Monitor",
            center=[0, exit_slit_distance, monochromator_z_offset],
        )

    def print_positions(self):
        print('#' * 20, self.name, '#' * 20)

        for element in (self.SuperCWiggler, self.FrontEnd,
                        self.MonochromatorCr1, self.Cr1Monitor, self.MonochromatorCr2, self.Cr2Monitor):
            print('#' * 5, element.name, 'at', element.center)

        for element in (self.MonochromatorCr1, self.MonochromatorCr2):
            print('#' * 5, element.name, 'RxRyRz', element.pitch, element.roll, element.yaw)

        print('#' * (42 + len(self.name)))


    def align_energy(self, en, d_en, mono=False, invert_croc=False):
        # changing energy for the beamline / source
        self.alignE = en
        if not mono:
            self.SuperCWiggler.eMin = en * (1. - d_en)
            self.SuperCWiggler.eMax = en * (1. + d_en)
        else:
            self.SuperCWiggler.eMin = en - 1.
            self.SuperCWiggler.eMax = en + 1.
        
        # re-making the CRL
        del self.CrocLensStack[:]
        if invert_croc:
            g_l, g_r = CrocLens.calc_y_g(lens_material, croc_crl_distance / 2., en, croc_crl_y_t, croc_crl_L), 0
        else:
            g_l, g_r = 0, CrocLens.calc_y_g(lens_material, croc_crl_distance / 2., en, croc_crl_y_t, croc_crl_L)
        
        
        self.CrocLensStack = CrocLens.make_stack(
            L=croc_crl_L, N=int(croc_crl_L), d=croc_crl_y_t, g_left=g_l, g_right=g_r,
            bl=self, 
            center=[0., croc_crl_distance, 0],
            material=lens_material,
            limPhysX=monochromator_x_lim, 
            limPhysY=monochromator_y_lim, 
        )
        
        # setting up pre-CRL mask
        apt = CrocLens.calc_optimal_params(lens_material, croc_crl_distance / 2., en)['Aperture']
        self.CrlMask.opening = [-100., 100., -apt / 2., apt / 2.]
        print('Croc Lens: g_r = %.01f, g_l = %.01f, y_t = %.01f, L = %.01f' % (g_r, g_l, croc_crl_y_t, croc_crl_L))
        print('Mask: %.01f' % apt)
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


    outDict = {
        'BeamSourceGlobal': beam_source,
        'BeamAperture1Local': beam_ap1,
    }
    
    # Diamond filters
    beamIn = beam_source
    for ifl, fl in enumerate(bl.FilterStackC):
        lglobal, llocal1, llocal2 = fl.double_refract(beam=beamIn)
        strl = '_{0:02d}'.format(ifl)
        outDict['BeamFilterCGlobal' + strl] = lglobal
        outDict['BeamFilterCLocal1' + strl] = llocal1
        outDict['BeamFilterCLocal2' + strl] = llocal2

        llocal2a = raycing.sources.Beam(copyFrom=llocal2)
        llocal2a.absorb_intensity(beamIn)
        outDict['BeamFilterCLocal2a' + strl] = llocal2a
        beamIn = lglobal
    
    # SiC filters
    for ifl, fl in enumerate(bl.FilterStackSiC):
        lglobal, llocal1, llocal2 = fl.double_refract(beam=beamIn)
        strl = '_{0:02d}'.format(ifl)
        outDict['BeamFilterSiCGlobal' + strl] = lglobal
        outDict['BeamFilterSiCLocal1' + strl] = llocal1
        outDict['BeamFilterSiCLocal2' + strl] = llocal2

        llocal2a = raycing.sources.Beam(copyFrom=llocal2)
        llocal2a.absorb_intensity(beamIn)
        outDict['BeamFilterSiCLocal2a' + strl] = llocal2a
        beamIn = lglobal
    
    # Pre-CRL mask
    outDict['BeamAperture2Local'] = bl.CrlMask.propagate(
        beam=beamIn
    )
    
    # CRL
    for ilens, lens in enumerate(bl.CrocLensStack):
        lglobal, llocal1, llocal2 = lens.double_refract(beamIn, needLocal=True)
        strl = '_{0:02d}'.format(ilens)
        outDict['BeamLensGlobal'+strl] = lglobal
        outDict['BeamLensLocal1'+strl] = llocal1
        outDict['BeamLensLocal2'+strl] = llocal2
        
        llocal2a = raycing.sources.Beam(copyFrom=llocal2)
        llocal2a.absorb_intensity(beamIn)
        outDict['BeamLensLocal2a'+strl] = llocal2a
        beamIn = lglobal

    beam_mono_c1_global, beam_mono_c1_local = bl.MonochromatorCr1.reflect(
        beam=beamIn
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

    outDict['BeamMonoC1Local'] = beam_mono_c1_local
    outDict['BeamMonoC1Global'] = beam_mono_c1_global
    outDict['BeamMonitor1Local'] = beam_mon1
    outDict['BeamMonoC2Local'] = beam_mono_c2_local
    outDict['BeamMonoC2Global'] = beam_mono_c2_global
    outDict['BeamMonitor2Local'] = beam_mon2

    bl.prepare_flow()

    return outDict

rrun.run_process = run_process

