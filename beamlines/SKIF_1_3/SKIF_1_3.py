import numpy as np

import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from params.sources import ring_kwargs, wiggler_1_3_kwargs
from params.params_1_3 import front_end_distance, front_end_h_angle, front_end_v_angle, filter_distance


# ################################################## SIM PARAMETERS ####################################################


show = True
repeats = 10


# ################################################# SETUP PARAMETERS ###################################################


""" Front End Aperture """
front_end_opening = [
    -front_end_distance * np.tan(front_end_h_angle / 2.),
    front_end_distance * np.tan(front_end_h_angle / 2.),
    -front_end_distance * np.tan(front_end_v_angle / 2.),
    front_end_distance * np.tan(front_end_v_angle / 2.)
]

""" Filter """
filter_thickness = 0.1


# #################################################### MATERIALS #######################################################


filterDiamond = rm.Material('C', rho=3.52, kind='plate')


# #################################################### BEAMLINE ########################################################


class SKIF13(raycing.BeamLine):
    def __init__(self):
        raycing.BeamLine.__init__(self)

        self.SuperCWiggler = rsources.Wiggler(
            name=r"Superconducting Wiggler",
            bl=self,
            center=[0, 0, 0],
            eMin=100,
            eMax=200000,
            **ring_kwargs,
            **wiggler_1_3_kwargs
        )

        self.FrontEnd = rapts.RectangularAperture(
            bl=self,
            name=r"Front End Slit",
            center=[0, front_end_distance, 0],
            opening=front_end_opening
        )

        self.FrontEndMonitor = rscreens.Screen(
            bl=self,
            name='Front End Monitor',
            center=[0, front_end_distance + 1, 0]
        )

        self.Filter = roes.Plate(
            bl=self,
            name='Diamond Window',
            center=[0, filter_distance, 0],
            pitch=.5 * np.pi,
            material=filterDiamond,
            t=filter_thickness
        )

        self.FilterMonitor = rscreens.Screen(
            bl=self,
            name='Front End Monitor',
            center=[0, filter_distance + 100, 0]
        )


# ################################################# BEAM TOPOLOGY ######################################################


def run_process(bl: SKIF13):
    beam_source = bl.sources[0].shine()

    _ = bl.FrontEnd.propagate(
        beam=beam_source
    )
    beam_monitor1 = bl.FrontEndMonitor.expose(
        beam=beam_source
    )
    beam_filter1_global, beam_filter1_local1, beam_filter1_local2 = bl.Filter.double_refract(
        beam=beam_source
    )
    beam_filter1_local2a = rsources.Beam(
        copyFrom=beam_filter1_local2
    )
    beam_filter1_local2a.absorb_intensity(
        inBeam=beam_source
    )
    beam_monitor2 = bl.FilterMonitor.expose(
        beam=beam_filter1_global
    )

    if show:
        bl.prepare_flow()

    return {
        'BeamSourceGlobal': beam_source,
        'BeamFrontEndMonitorLocal': beam_monitor1,
        'BeamFilter1Global': beam_filter1_global,
        'BeamFilter1Local1': beam_filter1_local1,
        'BeamFilter1Local2': beam_filter1_local2,
        'BeamFilter1Local2A': beam_filter1_local2a,
        'BeamFilterMonitorLocal': beam_monitor2,
    }


rrun.run_process = run_process


# ##################################################### PLOTS ##########################################################


plots = [
    xrtplot.XYCPlot(
        beam='BeamFrontEndMonitorLocal',
        title='Front End Intensity',
        xaxis=xrtplot.XYCAxis(r'$x$', 'mm', limits=(1.4 * front_end_opening[0], 1.4 * front_end_opening[1])),
        yaxis=xrtplot.XYCAxis(r'$z$', 'mm', limits=(1.4 * front_end_opening[2], 1.4 * front_end_opening[3])),
        ePos=1),
    xrtplot.XYCPlot(
        beam='BeamFrontEndMonitorLocal',
        title='Front End Power',
        fluxKind='power',
        xaxis=xrtplot.XYCAxis(r'$x$', 'mm', limits=(1.4 * front_end_opening[0], 1.4 * front_end_opening[1])),
        yaxis=xrtplot.XYCAxis(r'$z$', 'mm', limits=(1.4 * front_end_opening[2], 1.4 * front_end_opening[3])),
        ePos=1),
    xrtplot.XYCPlot(
        beam='BeamFilter1Local2A',
        title='Filter absorbed intensity',
        xaxis=xrtplot.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtplot.XYCAxis(r'$y$', 'mm'),
        caxis=xrtplot.XYCAxis('energy', 'keV'),
        fluxKind='total'),
    xrtplot.XYCPlot(
        beam='BeamFilter1Local2A',
        title='Filter absorbed power',
        xaxis=xrtplot.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtplot.XYCAxis(r'$y$', 'mm'),
        caxis=xrtplot.XYCAxis('energy', 'keV'),
        fluxKind='power',
        fluxFormatStr='%.0f'),
    xrtplot.XYCPlot(
        beam='BeamFilterMonitorLocal',
        title='Post-Filter Intensity',
        xaxis=xrtplot.XYCAxis(r'$x$', 'mm', limits=(5.4 * front_end_opening[0], 5.4 * front_end_opening[1])),
        yaxis=xrtplot.XYCAxis(r'$z$', 'mm', limits=(5.4 * front_end_opening[2], 5.4 * front_end_opening[3])),
        ePos=1),
]


# ############################################### SEQUENCE GENERATOR ###################################################


def none_scan(plts, bl: SKIF13):
    yield


# ###################################################### MAIN ##########################################################


if __name__ == '__main__':
    beamline = SKIF13()
    scan = none_scan

    if show:
        beamline.glow(
            scale=[1e3, 1e3, 1e3],
            centerAt=2,
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
