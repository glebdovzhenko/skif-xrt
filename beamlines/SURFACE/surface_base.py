import os
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials_elemental as rm
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.sources as rsources

from params.sources import ring_kwargs
from components import PrismaticLens


# ############################ SETUP PARAMETERS ###############################


# ################################ MATERIALS ##################################


# ################################ BEAMLINE ###################################
class SURFACE_BASE(raycing.BeamLine):
    """"""

    def __init__(self):
        raycing.BeamLine.__init__(self)
        self.name = "SURFACE"

        self.bm = rsources.BendingMagnet(
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

        self.CrocLensStack = []

        # 1:1@12.4 keV
        self.CrlEntranceMonitor = rscreens.Screen(
            bl=self, name="Source Monitor", center=[0.0, 20.998e3, 0.0]
        )
        self.CrlEntranceApt = rapts.RectangularAperture(
            bl=self, name="CRL Entrance Aperture", center=[0.0, 20.999e3, 0.0]
        )
        self.CrocLensStack = PrismaticLens.make_stack(
            L=90 * 3,
            N=90 * 3,
            d=1.2 * 3,
            # g_first=11.2,  # 5 keV
            # g_first=4.9,  # 7.5 keV
            g_first=2.75,  # 10 keV
            # g_first=1.751,  # 12.4 keV
            # g_first=1.2,  # 15 keV
            g_last=0.0,
            bl=self,
            center=[0.0, 21.0e3, 0],
            material=rm.Be(kind="lens"),
            limPhysX=[-70, 70],
            limPhysY=[-5, 5],
        )
        self.CrlMonitor = rscreens.Screen(
            bl=self, name="CRL Monitor", center=[0.0, 21.10e3, 0.0]
        )
        self.CrlFocusApt = rapts.RectangularAperture(
            bl=self, name="CRL Focus Aperture", center=[0.0, 41.990e3, 0.0]
        )
        self.CrlFocusMonitor = rscreens.Screen(
            bl=self, name="CRL Monitor", center=[0.0, 42.0e3, 0.0]
        )


# ############################# BEAM TOPOLOGY #################################
def run_process(bl: SURFACE_BASE):

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

    beam_focus_apt = bl.CrlFocusApt.propagate(beam=beamIn)
    outDict["BeamFocusAptLocal"] = beam_focus_apt

    beam_crl_focus = bl.CrlFocusMonitor.expose(beam=beamIn)
    outDict["BeamCRLFocusLocal"] = beam_crl_focus

    bl.prepare_flow()
    return outDict


rrun.run_process = run_process
