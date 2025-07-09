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
    7500: 5.45,
    10000: 2.85,
    12400: 1.92,
    15000: 1.3,
}


mono_r_s = {
    5000: 8150,
    7500: 5420,
    10000: 4070,
    12400: 3280,
    15000: 2710,
}

# ################################ MATERIALS ##################################
crystalSi01 = rmats.CrystalSi(rho=2.33, table=r"Chantler", name=None)


# ################################ BEAMLINE ###################################
class SURFACE_DIAGNOSTIC(raycing.BeamLine):
    """"""

    def __init__(self):
        raycing.BeamLine.__init__(self)
        self.name = "SURFACE VEIPS BRANCH"
        self.alignE = 15000

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
            **ring_kwargs,
        )

        self.SourceMonitor = rscreens.Screen(
            bl=self, name="Source Monitor", center=[0.0, 1.0, 0.0]
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
            opening=[-55, -11, -2, 2],
        )

        self.Mono = roes.DCMwithSagittalFocusing(
            bl=self,
            name="DCM",
            center=["auto", 24000, 0],
            yaw=1.5e-3,
            material=crystalSi01,
            material2=crystalSi01,
            Rs=mono_r_s[self.alignE],
            bragg="auto",
            fixedOffset=25.0,
            limPhysX=[-70, 70],
            limPhysY=[-30, 30],
        )

        self.CrlFocusApt = rapts.RectangularAperture(
            bl=self, name="BL Exit Aperture", center=["auto", 41.990e3, 25.0]
        )

        self.CrlFocusMonitor = rscreens.Screen(
            bl=self, name="BL Exit Monitor", center=["auto", 42.0e3, 25.0]
        )


# ############################# BEAM TOPOLOGY #################################
def run_process(bl: SURFACE_DIAGNOSTIC):

    beam_source = bl.sources[0].shine()
    beam_source_monitor = bl.SourceMonitor.expose(beam=beam_source)
    beam_crl_entrance = bl.CrlEntranceMonitor.expose(beam=beam_source)
    _ = bl.CrlEntranceApt.propagate(beam=beam_source)

    outDict = {
        "BeamSourceGlobal": beam_source,
        "BeamSourceLocal": beam_source_monitor,
        "BeamCRLEntranceLocal": beam_crl_entrance,
    }

    # CRL
    beamIn = beam_source
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
    outDict["BeamCRLExitLocal"] = beam_crl_exit

    outDict["BeamSplitterLocal"] = bl.Splitter.propagate(beam=beamIn)

    beam_mono_global, beam_mono_local, beam_mono_local2 = bl.Mono.double_reflect(
        beam=beamIn
    )
    outDict["BeamMonoGlobal"] = beam_mono_global
    outDict["BeamMonoLocal"] = beam_mono_local
    outDict["BeamMonoLocal2"] = beam_mono_local2

    beam_focus_apt = bl.CrlFocusApt.propagate(beam=beam_mono_global)
    outDict["BeamFocusAptLocal"] = beam_focus_apt

    beam_crl_focus = bl.CrlFocusMonitor.expose(beam=beam_mono_global)
    outDict["BeamCRLFocusLocal"] = beam_crl_focus

    bl.prepare_flow()
    return outDict


rrun.run_process = run_process
