import os
import numpy as np

import xrt.backends.raycing as raycing
from xrt.backends.raycing import materials
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
class LABTOMO(raycing.BeamLine):
    """"""

    def __init__(self):
        raycing.BeamLine.__init__(self)
        self.name = "Testron tomograph"

        self.gs = rsources.GeometricSource(
            name="Microfocus Tube",
            bl=self,
            center=[0, -15, 0],
            energies=(57.9e3,),  # W target
            distE="lines",
        )

        self.sample = roe.Plate(
            name="Sample",
            bl=self,
            center=[0, 0, 0],
            pitch=np.pi / 2,
            t=0.1,
            material=rm.Cu(kind="lens"),
        )

        self.detector = rscreens.Screen(name="Detector", bl=self, center=[0, 580, 0])


# ############################# BEAM TOPOLOGY #################################
def run_process(bl: LABTOMO):
    """"""
    beam_source = bl.sources[0].shine()
    beam_sample_global, beam_sample_local1, beam_sample_local2 = (
        bl.sample.double_refract(beam=beam_source)
    )
    beam_detector_local = bl.detector.expose(beam_sample_global)
    bl.prepare_flow()
    return {
        "BeamSourceGlobal": beam_source,
        "BeamSampleGlobal": beam_sample_global,
        "BeamSampleLocal1": beam_sample_local1,
        "BeamSampleLocal2": beam_sample_local2,
        "BeamDetectorLocal": beam_detector_local,
    }


rrun.run_process = run_process
