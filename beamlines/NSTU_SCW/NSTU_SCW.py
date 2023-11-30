import os
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.sources as rsources

from components import BentLaueParaboloid, CrystalSiPrecalc, PrismaticLens
from params.params_nstu_scw import (
    crl_mask_distance,
    croc_crl_L,
    croc_crl_distance,
    croc_crl_y_t,
    diamond_filter_N,
    diamond_filter_th,
    exit_slit_distance,
    filter_distance,
    front_end_distance,
    front_end_h_angle,
    front_end_opening,
    front_end_v_angle,
    monochromator_distance,
    monochromator_x_lim,
    monochromator_y_lim,
    monochromator_z_offset,
    sic_filter_N,
    sic_filter_th,
    filter_size_x,
    filter_size_z
)
from params.sources import ring_kwargs, wiggler_1_5_kwargs, wiggler_nstu_scw_kwargs
from utils.focus_locator import FocusLocator

# ############################ SETUP PARAMETERS ###############################


""" Monochromator """
monochromator_alignment_energy = 30.e3
monochromator_c1_alpha = np.radians(35.3)
monochromator_c1_thickness = .5
monochromator_c2_alpha = np.radians(35.3)
monochromator_c2_thickness = .5


# ################################ MATERIALS ##################################


cr_si_1 = CrystalSiPrecalc(
    hkl=(1, 1, 1),
    geom='Laue reflection',
    useTT=True,
    t=monochromator_c1_thickness,
    database=os.path.join(os.getenv('BASE_DIR', ''), 'components',
                          'Si111ref_sag.csv'))
cr_si_2 = CrystalSiPrecalc(
    hkl=(1, 1, 1),
    geom='Laue reflection',
    useTT=True,
    t=monochromator_c2_thickness,
    database=os.path.join(os.getenv('BASE_DIR', ''), 'components',
                          'Si111ref_sag.csv'))

mBeryllium = rm.Material('Be', rho=1.848, kind='lens')
mAl = rm.Material('Al', rho=2.7, kind='lens')
mDiamond = rm.Material('C', rho=3.5, kind='lens')
mGraphite = rm.Material('C', rho=2.15, kind='lens')
mGlassyCarbon = rm.Material('C', rho=1.50, kind='lens')
mDiamondF = rm.Material('C', rho=3.5)
mSiC = rm.Material(('Si', 'C'), quantities=(1, 1), rho=3.16)
lens_material = mGlassyCarbon


# ################################ BEAMLINE ###################################


FL = FocusLocator(
    beam_name='BeamMonoC2Global',
    data_dir=os.path.join(os.getenv('BASE_DIR', ''), 'datasets', 'tmp'),
    axes=['x', 'z'],
    z0=monochromator_z_offset,
    niterations=4
)


@FL.beamline
class NSTU_SCW(raycing.BeamLine):
    # metadata keys for organizing modelling results
    # these are variables that are potentially varied or scanned,
    # constants are stored elsewhere.
    md_keys = {
        'en': 'Beamline alignment energy, [eV]',
        'd_en': 'Source energy is in range [en * (1. - d_en), en * (1. - d_en)',
        'crl_mask_ox': 'CRL mask opening horizontal',
        'crl_mask_oz': 'CRL mask opening vertical',
        'crl_l': 'PrismaticLens length',
        'crl_d': 'PrismaticLens tooth height',
        'crl_g_first': 'PrismaticLens gap on entrance',
        'crl_g_last': 'PrismaticLens gap on exit',
        'crl_mat_name': 'PrismaticLens material name',
        'crl_mat_rho': 'PrismaticLens material density',
        'monoC1Rx': 'DCM 1st crystal Rx',
        'monoC1Ry': 'DCM 1st crystal Ry',
        'monoC2Rx': 'DCM 2nd crystal Rx',
        'monoC2Ry': 'DCM 2nd crystal Ry',
    }

    def __init__(self):
        raycing.BeamLine.__init__(self)

        self.name = "NSTU SCW"

        self._metadata = {k: None for k in self.md_keys.keys()}

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
                    center=[0, filter_distance + ii *
                            1.1 * diamond_filter_th, 0],
                    pitch=np.pi/2.,
                    material=mDiamondF,
                    t=diamond_filter_th,
                    limPhysX=[-filter_size_x / 2, filter_size_x / 2],
                    limPhysY=[-filter_size_z / 2, filter_size_z / 2]
                )
            )

        for ii in range(sic_filter_N):
            self.FilterStackSiC.append(
                roe.Plate(
                    name='SiC Filter %d' % (ii + 1),
                    bl=self,
                    center=[0, filter_distance + diamond_filter_N * 1.1 *
                            diamond_filter_th + ii * 1.1 * sic_filter_th, 0],
                    pitch=np.pi/2.,
                    material=mSiC,
                    t=sic_filter_th,
                    limPhysX=[-filter_size_x / 2, filter_size_x / 2],
                    limPhysY=[-filter_size_z / 2, filter_size_z / 2]
                )
            )

        self.CrlMask = rapts.RectangularAperture(
            bl=self,
            name=r"Front End Slit",
            center=[0, crl_mask_distance, 0],
            opening=front_end_opening
        )

        self.LensMaterial = lens_material
        self.CrocLensStack = PrismaticLens.make_stack(
            L=croc_crl_L,
            N=int(croc_crl_L),
            d=croc_crl_y_t,
            g_last=0.0,
            g_first=croc_crl_y_t,
            bl=self,
            center=[0., croc_crl_distance, 0],
            material=self.LensMaterial,
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

    def align_source(self, en, d_en, mono=False):
        self._metadata['en'] = en
        self._metadata['d_en'] = d_en
        self.alignE = en

        if not mono:
            self.SuperCWiggler.eMin = en * (1. - d_en)
            self.SuperCWiggler.eMax = en * (1. + d_en)
        else:
            self.SuperCWiggler.eMin = en - 1.
            self.SuperCWiggler.eMax = en + 1.

    def align_crl(self, L, N, d, g_f, g_l):
        self._metadata['crl_l'] = L
        self._metadata['crl_N'] = N
        self._metadata['crl_d'] = d
        self._metadata['crl_g_first'] = g_f
        self._metadata['crl_g_last'] = g_l
        self._metadata['crl_mat_name'] = self.LensMaterial.name
        self._metadata['crl_mat_rho'] = self.LensMaterial.rho


        del self.CrocLensStack[:]
        self.CrocLensStack = PrismaticLens.make_stack(
            L=L, N=N, d=d, g_first=g_f, g_last=g_l,
            bl=self,
            center=[0., croc_crl_distance, 0],
            material=self.LensMaterial,
            limPhysX=monochromator_x_lim,
            limPhysY=monochromator_y_lim,
        )

    def align_crl_mask(self, dx, dz):
        self._metadata['crl_mask_ox'] = dx
        self._metadata['crl_mask_oz'] = dz

        self.CrlMask.opening = [-dx/2., dx/2., -dz/2., dz/2.]

    def align_mono(self, en, R1x, R1y, R2x, R2y):
        self._metadata['monoC1Rx'] = R1x
        self._metadata['monoC1Ry'] = R1y
        self._metadata['monoC2Rx'] = R2x
        self._metadata['monoC2Ry'] = R2y

        self.MonochromatorCr1.Rx = R1x 
        self.MonochromatorCr1.Ry = R1y
        self.MonochromatorCr2.Rx = R2x
        self.MonochromatorCr2.Ry = R2y

        theta0 = np.arcsin(
            rm.ch / (2 * self.MonochromatorCr1.material[0].d * en))

        self.MonochromatorCr1.pitch = np.pi / 2 + theta0 + \
            self.MonochromatorCr1.alpha
        self.MonochromatorCr1.center = [0., monochromator_distance, 0.]

        self.MonochromatorCr2.pitch = np.pi / 2 - theta0 + \
            self.MonochromatorCr2.alpha
        self.MonochromatorCr2.center = [
            0.,
            monochromator_distance +
            monochromator_z_offset / np.tan(2. * theta0),
            monochromator_z_offset
        ]

        self.Cr1Monitor.center = [
            0.,
            monochromator_distance + .5 *
            monochromator_z_offset / np.tan(2. * theta0),
            .5 * monochromator_z_offset
        ]


# ############################# BEAM TOPOLOGY #################################


@FL.run_process
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

    # monochromator
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
