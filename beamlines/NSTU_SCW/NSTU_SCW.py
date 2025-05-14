import os
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.sources as rsources

from components import BentLaueParaboloid, PrismaticLens
from params.params_nstu_scw import (
    croc_crl_distance,
    croc_geometry_BSU,
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
    filter_size_z,
)
from params.sources import ring_kwargs, wiggler_1_5_kwargs


# ############################ SETUP PARAMETERS ###############################
""" Monochromator """
monochromator_alignment_energy = 30.0e3
monochromator_c1_alpha = np.radians(35.3)
monochromator_c1_thickness = 0.5
monochromator_c2_alpha = np.radians(35.3)
monochromator_c2_thickness = 0.5


# ################################ MATERIALS ##################################
cr_si_1 = rm.CrystalSi(
    hkl=(1, 1, 1),
    geom="Laue reflected",
    useTT=True,
    volumetricDiffraction=True,
    t=monochromator_c1_thickness,
    name=None,
)
cr_si_2 = rm.CrystalSi(
    hkl=(1, 1, 1),
    geom="Laue reflected",
    useTT=True,
    volumetricDiffraction=True,
    t=monochromator_c2_thickness,
    name=None,
)

mBeryllium = rm.Material("Be", rho=1.848, kind="lens")
mAl = rm.Material("Al", rho=2.7, kind="lens")
mDiamond = rm.Material("C", rho=3.5, kind="lens")
mGraphite = rm.Material("C", rho=2.15, kind="lens")
mGlassyCarbon = rm.Material("C", rho=1.50, kind="lens")
mDiamondF = rm.Material("C", rho=3.5, kind="lens")
mSiC = rm.Material(("Si", "C"), quantities=(1, 1), rho=3.16, kind="lens")
lens_material = mBeryllium


# ################################ BEAMLINE ###################################
class NSTU_SCW(raycing.BeamLine):
    """"""

    def __init__(self):
        raycing.BeamLine.__init__(self)
        self.name = "NSTU SCW"

        self.SuperCWiggler = rsources.Wiggler(
            name=r"Superconducting Wiggler",
            bl=self,
            center=[0, 0, 0],
            eMin=100,
            eMax=100000,
            xPrimeMax=front_end_h_angle * 0.505e3,
            zPrimeMax=front_end_v_angle * 0.505e3,
            **ring_kwargs,
            **wiggler_1_5_kwargs
        )

        self.SourceMonitor = rscreens.Screen(
            bl=self,
            name=r"Source Monitor",
            center=[0, wiggler_1_5_kwargs["period"] * wiggler_1_5_kwargs["n"] / 2.0, 0],
        )

        self.FrontEnd = rapts.RectangularAperture(
            bl=self,
            name=r"Front End Slit",
            center=[0, front_end_distance, 0],
            opening=front_end_opening,
        )

        self.FilterEffectiveC = None
        self.FilterEffectiveSiC = None
        self.FilterStackC = []
        self.FilterStackSiC = []

        self.set_filter_stacks()

        self.LensMaterial = lens_material
        self.CrocLensStack = []

        self.CrlMonitor = rscreens.Screen(
            bl=self,
            name=r"Lens Monitor",
            center=[0, croc_crl_distance, 0],
        )

        self.MonochromatorCr1 = roe.BentLaue2D(
            bl=self,
            name=r"Si[111] Crystal 1",
            center=[0.0, monochromator_distance, 0.0],
            pitch=np.pi / 2.0,
            roll=0.0,
            yaw=0.0,
            alpha=monochromator_c1_alpha,
            material=cr_si_1,
            targetOpenCL="CPU",
            limPhysY=monochromator_y_lim,
            limOptY=monochromator_y_lim,
            limPhysX=monochromator_x_lim,
            limOptX=monochromator_x_lim,
        )

        self.Cr1Monitor = rscreens.Screen(
            bl=self,
            name=r"Crystal 1 Monitor",
            center=[0, monochromator_distance, 0.5 * monochromator_z_offset],
        )

        self.MonochromatorCr2 = roe.BentLaue2D(
            bl=self,
            name=r"Si[111] Crystal 2",
            center=[0.0, monochromator_distance, monochromator_z_offset],
            positionRoll=np.pi,
            pitch=0.0,
            roll=0.0,
            yaw=0.0,
            alpha=monochromator_c2_alpha,
            material=cr_si_2,
            targetOpenCL="CPU",
            limPhysY=monochromator_y_lim,
            limOptY=monochromator_y_lim,
            limPhysX=monochromator_x_lim,
            limOptX=monochromator_x_lim,
        )

        self.ExitSlit = rapts.RectangularAperture(
            bl=self,
            name=r"Exit Slit",
            center=[0, exit_slit_distance, monochromator_z_offset],
            opening=front_end_opening,
        )

    def set_effective_filters(self):
        """
        Replaces filter stack with one plate per material with the same total
        thickness.
        """
        self.FilterEffectiveC = roe.Plate(
            name="Diamond Filter",
            bl=self,
            center=[0, filter_distance, 0],
            pitch=np.pi / 2.0,
            material=mDiamondF,
            t=diamond_filter_th * diamond_filter_N,
            limPhysX=[-filter_size_x / 2, filter_size_x / 2],
            limPhysY=[-filter_size_z / 2, filter_size_z / 2],
        )
        self.FilterEffectiveSiC = roe.Plate(
            name="SiC Filter",
            bl=self,
            center=[
                0,
                filter_distance + diamond_filter_N * 1.1 * diamond_filter_th,
                0,
            ],
            pitch=np.pi / 2.0,
            material=mSiC,
            t=sic_filter_th * sic_filter_N,
            limPhysX=[-filter_size_x / 2, filter_size_x / 2],
            limPhysY=[-filter_size_z / 2, filter_size_z / 2],
        )
        del self.FilterStackC[:]
        del self.FilterStackSiC[:]

    def set_filter_stacks(self):
        self.FilterEffectiveC = None
        self.FilterEffectiveSiC = None

        for ii in range(diamond_filter_N):
            self.FilterStackC.append(
                roe.Plate(
                    name="Diamond Filter %d" % (ii + 1),
                    bl=self,
                    center=[0, filter_distance + ii * 1.1 * diamond_filter_th, 0],
                    pitch=np.pi / 2.0,
                    material=mDiamondF,
                    t=diamond_filter_th,
                    limPhysX=[-filter_size_x / 2, filter_size_x / 2],
                    limPhysY=[-filter_size_z / 2, filter_size_z / 2],
                )
            )

        for ii in range(sic_filter_N):
            self.FilterStackSiC.append(
                roe.Plate(
                    name="SiC Filter %d" % (ii + 1),
                    bl=self,
                    center=[
                        0,
                        filter_distance
                        + diamond_filter_N * 1.1 * diamond_filter_th
                        + ii * 1.1 * sic_filter_th,
                        0,
                    ],
                    pitch=np.pi / 2.0,
                    material=mSiC,
                    t=sic_filter_th,
                    limPhysX=[-filter_size_x / 2, filter_size_x / 2],
                    limPhysY=[-filter_size_z / 2, filter_size_z / 2],
                )
            )

    def align_source(self, en, d_en=None):
        """
        Set Wiggler photon energy range.
        default d_en=None is 1 eV
        """
        if d_en is not None:
            self.SuperCWiggler.eMin = en - d_en
            self.SuperCWiggler.eMax = en + d_en
        else:
            self.SuperCWiggler.eMin = en - 0.5
            self.SuperCWiggler.eMax = en + 0.5

    def align_front_end(self, dxprime=front_end_h_angle, dzprime=front_end_v_angle):
        """ """
        self.FrontEnd.opening = [
            -front_end_distance * np.tan(dxprime / 2.0),
            front_end_distance * np.tan(dxprime / 2.0),
            -front_end_distance * np.tan(dzprime / 2.0),
            front_end_distance * np.tan(dzprime / 2.0),
        ]
        self.SuperCWiggler.xPrimeMax = 1e3 * dxprime / 2.0
        self.SuperCWiggler.zPrimeMax = 1e3 * dzprime / 2.0

    def align_crl(self, L, N, d, g_f, g_l):
        del self.CrocLensStack[:]
        self.CrocLensStack = PrismaticLens.make_stack(
            L=L,
            N=N,
            d=d,
            g_first=g_f,
            g_last=g_l,
            bl=self,
            center=[0.0, croc_crl_distance, 0],
            material=self.LensMaterial,
            limPhysX=monochromator_x_lim,
            limPhysY=monochromator_y_lim,
        )

        self.CrlMonitor.center = [0, self.CrocLensStack[-1].center[1] + 10, 0]

    def align_mono(self, en, R1x, R1y, R2x, R2y, c1_en_offset=0.0, c2_en_offset=0.0):
        self.MonochromatorCr1.Rx = R1x
        self.MonochromatorCr1.Ry = R1y
        self.MonochromatorCr2.Rx = R2x
        self.MonochromatorCr2.Ry = R2y

        theta0 = np.arcsin(
            rm.ch / (2 * self.MonochromatorCr1.material[0].d * (en + c1_en_offset))
        )
        print("#####################################", np.degrees(theta0))
        self.MonochromatorCr1.pitch = np.pi / 2 + theta0 + self.MonochromatorCr1.alpha
        self.MonochromatorCr1.center = [0.0, monochromator_distance, 0.0]

        theta0 = np.arcsin(
            rm.ch / (2 * self.MonochromatorCr1.material[0].d * (en + c2_en_offset))
        )

        self.MonochromatorCr2.pitch = np.pi / 2 - theta0 + self.MonochromatorCr2.alpha
        self.MonochromatorCr2.center = [
            0.0,
            monochromator_distance + monochromator_z_offset / np.tan(2.0 * theta0),
            monochromator_z_offset,
        ]

        self.Cr1Monitor.center = [
            0.0,
            monochromator_distance
            + 0.5 * monochromator_z_offset / np.tan(2.0 * theta0),
            0.5 * monochromator_z_offset,
        ]

    def align_30_keV(self):
        """
        Assuming Rm / Rs = 6.
        """
        self.SuperCWiggler.xPrimeMax = 1.0
        self.SuperCWiggler.zPrimeMax = 0.1
        self.SuperCWiggler.eMin = 29950.0
        self.SuperCWiggler.eMax = 30050
        self.SuperCWiggler.eN = 101

        self.MonochromatorCr1.center = [0.0, 33000.0, 0.0]
        self.MonochromatorCr1.pitch = 2.252850
        self.MonochromatorCr1.alpha = np.radians(35.3)
        self.MonochromatorCr1.Rm = 2060.0 * 6
        self.MonochromatorCr1.Rs = -2060.0

        self.MonochromatorCr2.center = [0.0, 33189.355, 25.0]
        self.MonochromatorCr2.pitch = 2.120950
        self.MonochromatorCr2.alpha = np.radians(35.3)
        self.MonochromatorCr2.positionRoll = np.pi
        self.MonochromatorCr2.Rm = 2060.0 * 6
        self.MonochromatorCr2.Rs = -2060.0

        self.Cr1Monitor.center = [
            0.0,
            33094.678,
            12.5,
        ]

        self.LensMaterial = mBeryllium
        self.align_crl(
            L=croc_geometry_BSU["Be"]["L"],
            N=croc_geometry_BSU["Be"]["N"],
            d=croc_geometry_BSU["Be"]["y_t"],
            g_f=1.587,
            g_l=0.0,
        )

        self.align_front_end(
            dzprime=2.0
            * croc_geometry_BSU["Be"]["y_t"]
            / self.CrocLensStack[0].center[1]
        )

    def align_30_keV_2(self):
        """
        Assuming Rm / Rs = 12. f croc = 8150
        """
        self.SuperCWiggler.xPrimeMax = 1.0
        self.SuperCWiggler.zPrimeMax = 0.1
        self.SuperCWiggler.eMin = 29950.0
        self.SuperCWiggler.eMax = 30050
        self.SuperCWiggler.eN = 101

        self.MonochromatorCr1.center = [0.0, 33000.0, 0.0]
        self.MonochromatorCr1.pitch = 2.252815
        self.MonochromatorCr1.alpha = np.radians(35.3)
        self.MonochromatorCr1.Rm = 2060.0 * 12
        self.MonochromatorCr1.Rs = -2060.0

        self.MonochromatorCr2.center = [0.0, 33189.355, 25.0]
        self.MonochromatorCr2.pitch = 2.120988
        self.MonochromatorCr2.alpha = np.radians(35.3)
        self.MonochromatorCr2.positionRoll = np.pi
        self.MonochromatorCr2.Rm = 2060.0 * 12
        self.MonochromatorCr2.Rs = -2060.0

        self.Cr1Monitor.center = [
            0.0,
            33094.678,
            12.5,
        ]

        self.LensMaterial = mBeryllium
        self.align_crl(
            L=croc_geometry_BSU["Be"]["L"],
            N=croc_geometry_BSU["Be"]["N"],
            d=croc_geometry_BSU["Be"]["y_t"],
            g_f=1.587,
            g_l=0.0,
        )

        self.align_front_end(
            dzprime=2.0
            * croc_geometry_BSU["Be"]["y_t"]
            / self.CrocLensStack[0].center[1]
        )


# ############################# BEAM TOPOLOGY #################################


def run_process(bl: NSTU_SCW):

    beam_source = bl.sources[0].shine()
    beam_source_monitor = bl.SourceMonitor.expose(beam=beam_source)

    beam_ap1 = bl.FrontEnd.propagate(beam=beam_source)

    outDict = {
        "BeamSourceGlobal": beam_source,
        "BeamSourceLocal": beam_source_monitor,
        "BeamAperture1Local": beam_ap1,
    }
    beamIn = beam_source

    # Diamond filters
    if bl.FilterEffectiveC is not None:
        lglobal, llocal1, llocal2 = bl.FilterEffectiveC.double_refract(beam=beamIn)
        outDict["BeamFilterCGlobal"] = lglobal
        outDict["BeamFilterCLocal1"] = llocal1
        outDict["BeamFilterCLocal2"] = llocal2

        llocal2a = raycing.sources.Beam(copyFrom=llocal2)
        llocal2a.absorb_intensity(beamIn)
        outDict["BeamFilterCLocal2a"] = llocal2a
        beamIn = lglobal
    else:
        for ifl, fl in enumerate(bl.FilterStackC):
            lglobal, llocal1, llocal2 = fl.double_refract(beam=beamIn)
            strl = "_{0:02d}".format(ifl)
            outDict["BeamFilterCGlobal" + strl] = lglobal
            outDict["BeamFilterCLocal1" + strl] = llocal1
            outDict["BeamFilterCLocal2" + strl] = llocal2

            llocal2a = raycing.sources.Beam(copyFrom=llocal2)
            llocal2a.absorb_intensity(beamIn)
            outDict["BeamFilterCLocal2a" + strl] = llocal2a
            beamIn = lglobal

    # SiC filters
    if bl.FilterEffectiveSiC is not None:
        lglobal, llocal1, llocal2 = bl.FilterEffectiveSiC.double_refract(beam=beamIn)
        outDict["BeamFilterSiCGlobal"] = lglobal
        outDict["BeamFilterSiCLocal1"] = llocal1
        outDict["BeamFilterSiCLocal2"] = llocal2

        llocal2a = raycing.sources.Beam(copyFrom=llocal2)
        llocal2a.absorb_intensity(beamIn)
        outDict["BeamFilterSiCLocal2a"] = llocal2a
        beamIn = lglobal
    else:
        for ifl, fl in enumerate(bl.FilterStackSiC):
            lglobal, llocal1, llocal2 = fl.double_refract(beam=beamIn)
            strl = "_{0:02d}".format(ifl)
            outDict["BeamFilterSiCGlobal" + strl] = lglobal
            outDict["BeamFilterSiCLocal1" + strl] = llocal1
            outDict["BeamFilterSiCLocal2" + strl] = llocal2

            llocal2a = raycing.sources.Beam(copyFrom=llocal2)
            llocal2a.absorb_intensity(beamIn)
            outDict["BeamFilterSiCLocal2a" + strl] = llocal2a
            beamIn = lglobal

    # # CRL
    for ilens, lens in enumerate(bl.CrocLensStack):
        lglobal, llocal1, llocal2 = lens.double_refract(beamIn, needLocal=True)
        strl = "_{0:02d}".format(ilens)
        outDict["BeamLensGlobal" + strl] = lglobal
        outDict["BeamLensLocal1" + strl] = llocal1
        outDict["BeamLensLocal2" + strl] = llocal2

        llocal2a = raycing.sources.Beam(copyFrom=llocal2)
        llocal2a.absorb_intensity(beamIn)
        outDict["BeamLensLocal2a" + strl] = llocal2a
        beamIn = lglobal

    beam_crl_exit = bl.CrlMonitor.expose(beam=beamIn)

    # monochromator
    beam_mono_c1_global, beam_mono_c1_local = bl.MonochromatorCr1.reflect(beam=beamIn)

    beam_mon1 = bl.Cr1Monitor.expose(beam=beam_mono_c1_global)

    beam_mono_c2_global, beam_mono_c2_local = bl.MonochromatorCr2.reflect(
        beam=beam_mono_c1_global
    )

    beam_mon2 = bl.ExitSlit.propagate(beam=beam_mono_c2_global)

    outDict["BeamLensExitLocal"] = beam_crl_exit
    outDict["BeamMonoC1Local"] = beam_mono_c1_local
    outDict["BeamMonoC1Global"] = beam_mono_c1_global
    outDict["BeamMonitor1Local"] = beam_mon1
    outDict["BeamMonoC2Local"] = beam_mono_c2_local
    outDict["BeamMonoC2Global"] = beam_mono_c2_global
    outDict["BeamMonitor2Local"] = beam_mon2

    bl.prepare_flow()

    return outDict


rrun.run_process = run_process
