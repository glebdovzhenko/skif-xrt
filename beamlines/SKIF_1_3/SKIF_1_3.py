import numpy as np

import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
from components import PrismaticLens

from params.sources import ring_kwargs, wiggler_1_3_kwargs
from params.params_1_3 import front_end_distance, front_end_h_angle, front_end_v_angle
from params.params_1_3 import (
    croc_1_distance,
    # croc_2_distance,
    # sample_1_distance,
    sample_2_distance,
)


# ############################ SETUP PARAMETERS ###############################


crl_pos = croc_1_distance
sample_pos = sample_2_distance


# ############################### MATERIALS ###################################


mBeryllium = rm.Material("Be", rho=1.848, kind="lens")
mAl = rm.Material("Al", rho=2.7, kind="lens")
mGlassyCarbon = rm.Material("C", rho=1.50, kind="lens")
lens_material = mGlassyCarbon


# ############################### BEAMLINE ####################################


class SKIF13(raycing.BeamLine):
    def __init__(self):
        raycing.BeamLine.__init__(self)

        self.SuperCWiggler = rsources.Wiggler(
            name=r"Superconducting Wiggler",
            bl=self,
            center=[0, 0, 0],
            # eMin=100,
            # eMax=100e3,
            eMin=35e3 - 1.0,
            eMax=35e3 + 1.0,
            xPrimeMax=front_end_h_angle * 0.505e3,
            zPrimeMax=front_end_v_angle * 0.505e3,
            uniformRayDensity=True,
            **ring_kwargs,
            **wiggler_1_3_kwargs
        )

        self.WigglerMonitor = rscreens.Screen(
            bl=self,
            name=r"SCWg Monitor",
            center=[0, front_end_distance - 100, 0],
        )

        self.FrontEnd = rapts.RectangularAperture(
            bl=self,
            name=r"Front End Slit",
            center=[0, front_end_distance, 0],
            opening=[
                -front_end_distance * np.tan(front_end_h_angle / 2.0),
                front_end_distance * np.tan(front_end_h_angle / 2.0),
                -front_end_distance * np.tan(front_end_v_angle / 2.0),
                front_end_distance * np.tan(front_end_v_angle / 2.0),
            ],
        )

        self.FrontEndMonitor = rscreens.Screen(
            bl=self,
            name=r"FE Monitor",
            center=[0, front_end_distance + 100, 0],
        )

        self.LensEntranceMonitor = rscreens.Screen(
            bl=self,
            name=r"Lens Entrance Monitor",
            center=[0, crl_pos - 100, 0],
        )

        self.LensMaterial = lens_material
        self.lens_pars = PrismaticLens.calc_optimal_params(
            mat=lens_material,
            fdist=1.0 / (1.0 / crl_pos + 1.0 / (sample_pos - crl_pos)),
            en=35e3,
        )
        self.CrocLensStack = PrismaticLens.make_stack(
            L=self.lens_pars["L"],
            N=int(self.lens_pars["L"]),
            d=self.lens_pars["y_t"],
            g_last=0.0,
            g_first=self.lens_pars["y_t"],
            bl=self,
            center=[0.0, crl_pos, 0],
            material=self.LensMaterial,
            limPhysX=[
                -crl_pos * np.tan(front_end_h_angle / 2.0),
                crl_pos * np.tan(front_end_h_angle / 2.0),
            ],
            limPhysY=[
                -crl_pos * np.tan(front_end_v_angle / 2.0),
                crl_pos * np.tan(front_end_v_angle / 2.0),
            ],
        )

        self.LensExitMonitor = rscreens.Screen(
            bl=self,
            name=r"Lens Exit Monitor",
            center=[0, crl_pos + self.lens_pars["L"] + 100, 0],
        )

        self.SampleSlit = rapts.RectangularAperture(
            bl=self,
            name=r"Sample Slit",
            center=[0, sample_pos - 1, 0],
            opening=[
                -sample_pos * np.tan(front_end_h_angle / 2.0),
                sample_pos * np.tan(front_end_h_angle / 2.0),
                -sample_pos * np.tan(front_end_v_angle / 2.0),
                sample_pos * np.tan(front_end_v_angle / 2.0),
            ],
        )

        self.SampleMonitor = rscreens.Screen(
            bl=self,
            name=r"Sample Monitor",
            center=[0, sample_pos, 0],
        )


# ############################ BEAM TOPOLOGY ##################################


def run_process(bl: SKIF13):
    # Generating source beams
    outDict = {"SourceGlobal": bl.sources[0].shine()}

    # Exposing Wiggler Monitor
    outDict["WgMonitorLocal"] = bl.WigglerMonitor.expose(beam=outDict["SourceGlobal"])

    # Propagating through front-end
    outDict["Ap1Local"] = bl.FrontEnd.propagate(beam=outDict["SourceGlobal"])

    # Exposing Front-End Monitor
    outDict["FEMonitorLocal"] = bl.FrontEndMonitor.expose(beam=outDict["SourceGlobal"])

    # Exposing CRL Entrance Monitor
    outDict["LensEntranceMonitorLocal"] = bl.LensEntranceMonitor.expose(
        beam=outDict["SourceGlobal"]
    )

    # CRL
    beamIn = outDict["SourceGlobal"]
    for ilens, lens in enumerate(bl.CrocLensStack):
        lglobal, llocal1, llocal2 = lens.double_refract(beamIn, needLocal=True)
        strl = "_{0:02d}".format(ilens)
        outDict["LensGlobal" + strl] = lglobal
        outDict["LensLocal1" + strl] = llocal1
        outDict["LensLocal2" + strl] = llocal2

        llocal2a = raycing.sources.Beam(copyFrom=llocal2)
        llocal2a.absorb_intensity(beamIn)
        outDict["LensLocal2a" + strl] = llocal2a
        beamIn = lglobal

    # Exposing CRL Exit Monitor
    outDict["LensExitMonitorLocal"] = bl.LensExitMonitor.expose(beam=beamIn)

    # Propagating through sample slit
    outDict["SampleSlitLocal"] = bl.SampleSlit.propagate(beam=beamIn)

    # Exposing CRL Exit Monitor
    outDict["SampleMonitorLocal"] = bl.SampleMonitor.expose(beam=beamIn)

    bl.prepare_flow()

    return outDict


rrun.run_process = run_process
