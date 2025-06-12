import os
import numpy as np

import xrt.backends.raycing as raycing
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials_elemental as rm
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.sources as rsources

from params.sources import ring_kwargs
from components import PrismaticLens


# ############################ SETUP PARAMETERS ###############################


# ################################ MATERIALS ##################################


# ################################ BEAMLINE ###################################
class SURFACE(raycing.BeamLine):
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
            **ring_kwargs
        )

        self.SourceMonitor = rscreens.Screen(
            bl=self, name="Source Monitor", center=[0.0, 1.0, 0.0]
        )

        self.CrlEntranceMonitor = rscreens.Screen(
            bl=self, name="Source Monitor", center=[0.0, 10.0e3, 0.0]
        )

        self.CrocLensStack = []
        # 1:1@12.4 keV
        # self.CrocLensStack = PrismaticLens.make_stack(
        #     L=80,
        #     N=80,
        #     d=0.8,
        #     g_first=1.125,
        #     g_last=0.0,
        #     bl=self,
        #     center=[0.0, 10e3, 0],
        #     material=rm.Be(kind="lens"),
        #     limPhysX=[-70, 70],
        #     limPhysY=[-5, 5],
        # )

        # 1:3@12.4
        self.CrocLensStack = PrismaticLens.make_stack(
            L=90,
            N=90,
            d=1.0,
            g_first=1.53,
            g_last=0.0,
            bl=self,
            center=[0.0, 10e3, 0],
            material=rm.Be(kind="lens"),
            limPhysX=[-70, 70],
            limPhysY=[-5, 5],
        )

        # rotate lenses for horizontal focusing
        # for lens in self.CrocLensStack:
        #     lens.roll = np.pi / 2

        self.CrlMonitor = rscreens.Screen(
            bl=self, name="CRL Monitor", center=[0.0, 10.10e3, 0.0]
        )

        # 1:1
        # self.CrlFocusApt = rapts.RectangularAperture(
        #     bl=self, name="CRL Focus Aperture", center=[0.0, 19.990e3, 0.0]
        # )
        # self.CrlFocusMonitor = rscreens.Screen(
        #     bl=self, name="CRL Monitor", center=[0.0, 20.0e3, 0.0]
        # )

        # 1:3
        self.CrlFocusApt = rapts.RectangularAperture(
            bl=self, name="CRL Focus Aperture", center=[0.0, 39.990e3, 0.0]
        )
        self.CrlFocusMonitor = rscreens.Screen(
            bl=self, name="CRL Monitor", center=[0.0, 40.0e3, 0.0]
        )


# ############################# BEAM TOPOLOGY #################################
def run_process(bl: SURFACE):

    beam_source = bl.sources[0].shine()
    beam_source_monitor = bl.SourceMonitor.expose(beam=beam_source)
    beam_crl_entrance = bl.CrlEntranceMonitor.expose(beam=beam_source)

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
