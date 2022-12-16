import numpy as np
from copy import deepcopy

# import matplotlib
# matplotlib.use('agg')

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

from params.sources import ring_kwargs, undulator_1_1_kwargs
from params.params_1_1 import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle


# ################################################## SIM PARAMETERS ####################################################


show = True
repeats = 1
scan = 'energy_scan'


# ################################################# SETUP PARAMETERS ###################################################


# #################################################### MATERIALS #######################################################


# #################################################### BEAMLINE ########################################################


class SKIF11(raycing.BeamLine):
    def __init__(self):
        raycing.BeamLine.__init__(self)

        self.name = r"SKIF 1-1"

        self.alignE = 25.e3

        self.SuperCUndulator = rsources.Undulator(
            name=r"Superconducting Undulator",
            bl=self,
            center=[0, 0, 0],
            eMin=100,
            eMax=35000,
            xPrimeMax=front_end_h_angle * .505e3,
            zPrimeMax=front_end_v_angle * .505e3,
            nrays=10000,
            # xPrimeMaxAutoReduce=False,
            # zPrimeMaxAutoReduce=False,
            **ring_kwargs,
            **undulator_1_1_kwargs
        )

        self.FrontEnd = rapts.RectangularAperture(
            bl=self,
            name=r"Front End Slit",
            center=[0, front_end_distance, 0],
            opening=front_end_opening
        )


# ################################################# BEAM TOPOLOGY ######################################################


def run_process(bl: SKIF11):

    beam_source = bl.sources[0].shine()

    beam_ap1 = bl.FrontEnd.propagate(
        beam=beam_source
    )

    if show:
        bl.prepare_flow()

    return {
        'BeamSourceGlobal': beam_source,
        'BeamAperture1Local': beam_ap1,
    }


rrun.run_process = run_process


# ##################################################### PLOTS ##########################################################


plots = [
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title='FE-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title="FE-XprZpr",
        xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
]


# ############################################### SEQUENCE GENERATOR ###################################################


def print_positions(bl: SKIF11):
    print('#' * 20, bl.name, '#' * 20)

    for element in (bl.SuperCUndulator, bl.FrontEnd):
        print('#' * 5, element.name, 'at', element.center)

    print('#' * (42 + len(bl.name)))


def energy_scan(plts, bl: SKIF11):
    print_positions(bl)
    yield


# ###################################################### MAIN ##########################################################


if __name__ == '__main__':
    beamline = SKIF11()
    scan = vars()[scan]

    if show:
        beamline.glow(
            scale=[1e3, 1e3, 1e3],
            centerAt=r'Si[111] Crystal 1',
            generator=scan,
            generatorArgs=[plots, beamline],
            startFrom=1
        )
    else:
        xrtrun.run_ray_tracing(
            beamLine=beamline,
            plots=plots,
            repeats=repeats,
            backend=r"raycing",
            generator=scan,
            generatorArgs=[plots, beamline]
        )
