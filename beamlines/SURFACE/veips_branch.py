import os
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.materials_elemental as rm
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.sources as rsources

from params.sources import ring_kwargs
from components import PrismaticLens


# ############################ SETUP PARAMETERS ###############################
crl_g_first = {
    5000: 14.6,
    7500: 5.45,  # optimized
    10000: 2.85,  # optimized
    12400: 1.92,
    15000: 1.3,
}

mono_r_m = {
    5000: 63400,
    7500: 84800,
    10000: 109000,
    12400: 133000,
    15000: 159000,
}

# ################################ MATERIALS ##################################
crystalSi01 = rmats.CrystalSi(
    rho=2.33, geom="Bragg reflected", table=r"Chantler", name=None, useTT=True
)


# ################################ BEAMLINE ###################################
class SURFACE_VEIPS(raycing.BeamLine):
    """"""

    def __init__(self):
        raycing.BeamLine.__init__(self)
        self.name = "SURFACE VEIPS BRANCH"
        self.alignE = 12400

        r_k = ring_kwargs.copy()
        r_k["betaX"] = 0.252
        r_k["betaZ"] = 7.77
        self.bm = rsources.BendingMagnet(
            nrays=100000,
            name="Bending Magnet",
            bl=self,
            center=[0.0, 0.0, 0.0],
            eMin=5e3,
            eMax=15e3,
            xPrimeMax=2.50,
            zPrimeMax=2.0,
            B0=2.0,
            **r_k,
        )

        self.SourceMonitor = rscreens.Screen(
            bl=self, name="Source Monitor", center=[0.0, 20.0, 0.0]
        )

        self.Filter1 = roes.Plate(
            bl=self,
            name="Filter 1",
            center=[0.0, 11000.0, 0.0],
            pitch=np.pi / 2,
            material=rm.Be(kind="lens"),
            t=0.3,
        )

        self.Filter2 = roes.Plate(
            bl=self,
            name="Filter 2",
            center=[0.0, 13000.0, 0.0],
            pitch=np.pi / 2,
            material=rm.Be(kind="lens"),
            t=0.3,
        )

        self.CrlEntranceMonitor = rscreens.Screen(
            bl=self, name="CRL Entrance Monitor", center=[0.0, 20.998e3, 0.0]
        )

        self.CrlEntranceApt = rapts.RectangularAperture(
            bl=self, name="CRL Entrance Aperture", center=[0.0, 20.999e3, 0.0]
        )

        self.CrocLensStack = PrismaticLens.make_stack(
            L=90 * 3,
            N=90 * 3,
            d=1.2 * 3,
            g_first=crl_g_first[self.alignE],
            g_last=0.0,
            bl=self,
            center=[0.0, 21.0e3, 0],
            material=rm.Be(kind="lens"),
            limPhysX=[-70, 70],
            limPhysY=[-5, 5],
        )

        self.CrlMonitor = rscreens.Screen(
            bl=self, name="CRL Exit Monitor", center=[0.0, 21.30e3, 0.0]
        )

        self.Splitter = rapts.RectangularAperture(
            bl=self,
            name="Beam splitter",
            center=[0, 22000, 0],
            opening=[11, 55, -2, 2],
        )

        self.Mono = roes.JohannCylinder(
            bl=self,
            name="SCM",
            center=[r"auto", 24000, 0],
            pitch=r"auto",
            roll=np.pi / 2,
            material=crystalSi01,
            alpha=0,
            limPhysY=[-150.0, 150.0],
            Rm=mono_r_m[self.alignE],  # 10 keV
            targetOpenCL=[2, 0],
        )

        self.MonoAbsorber = roes.Plate(
            name="Si Absorber Plate",
            bl=self,
            center=[r"auto", 24000, 0],
            pitch=np.pi / 18,
            roll=np.pi / 2,
            material=rm.Si(kind="lens"),
            t=1.0,
            limPhysX=[-10, 10],
            limPhysY=[-150, 150],
        )

        self.CrlFocusApt = rapts.RectangularAperture(
            bl=self, name="BL Exit Aperture", center=[r"auto", 41.990e3, 0.0]
        )

        self.CrlFocusMonitor = rscreens.Screen(
            bl=self, name="BL Exit Monitor", center=[r"auto", 42.0e3, 0.0]
        )


# ############################# BEAM TOPOLOGY #################################
def run_process(bl: SURFACE_VEIPS):

    beam_source = bl.sources[0].shine()
    beam_source_monitor = bl.SourceMonitor.expose(beam=beam_source)

    beam_f1_global, beam_f1_local1, beam_f1_local2 = bl.Filter1.double_refract(
        beam=beam_source, returnLocalAbsorbed=0
    )

    beam_f2_global, beam_f2_local1, beam_f2_local2 = bl.Filter2.double_refract(
        beam=beam_f1_global, returnLocalAbsorbed=0
    )

    beam_crl_entrance = bl.CrlEntranceMonitor.expose(beam=beam_f2_global)
    _ = bl.CrlEntranceApt.propagate(beam=beam_f2_global)

    outDict = {
        "BeamSourceGlobal": beam_source,
        "BeamSourceLocal": beam_source_monitor,
        "BeamCRLEntranceLocal": beam_crl_entrance,
        "BeamFilter1Global": beam_f1_global,
        "BeamFilter1Local1": beam_f1_local1,
        "BeamFilter1Local2": beam_f1_local2,
        "BeamFilter2Global": beam_f2_global,
        "BeamFilter2Local1": beam_f2_local1,
        "BeamFilter2Local2": beam_f2_local2,
    }

    # CRL
    beamIn = beam_f2_global
    for ilens, lens in enumerate(bl.CrocLensStack):
        lglobal, llocal1, llocal2 = lens.double_refract(
            beamIn, needLocal=True, returnLocalAbsorbed=0
        )
        strl = "_{0:02d}".format(ilens)
        outDict["BeamLensGlobal" + strl] = lglobal
        outDict["BeamLensLocal1" + strl] = llocal1
        outDict["BeamLensLocal2" + strl] = llocal2

        beamIn = lglobal

    beam_crl_exit = bl.CrlMonitor.expose(beam=beamIn)
    outDict["BeamCRLExitLocal"] = beam_crl_exit

    outDict["BeamSplitterLocal"] = bl.Splitter.propagate(beam=beamIn)

    if False:
        beam_mono_global, beam_mono_local, beam_mono_local2 = (
            bl.MonoAbsorber.double_refract(beam=beamIn, returnLocalAbsorbed=0)
        )
        outDict["BeamMonoGlobal"] = beam_mono_global
        outDict["BeamMonoLocal"] = beam_mono_local
        outDict["BeamMonoLocal2"] = beam_mono_local2

    else:
        beam_mono_global, beam_mono_local = bl.Mono.reflect(beam=beamIn)
        outDict["BeamMonoGlobal"] = beam_mono_global
        outDict["BeamMonoLocal"] = beam_mono_local

        beam_focus_apt = bl.CrlFocusApt.propagate(beam=beam_mono_global)
        outDict["BeamFocusAptLocal"] = beam_focus_apt

        beam_crl_focus = bl.CrlFocusMonitor.expose(beam=beam_mono_global)
        outDict["BeamCRLFocusLocal"] = beam_crl_focus

    bl.prepare_flow()
    return outDict


rrun.run_process = run_process
