import numpy as np
from copy import deepcopy

import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from params.sources import ring_kwargs, wiggler_1_5_kwargs
from params.params_1_5 import front_end_distance, front_end_h_angle, front_end_v_angle, monochromator_distance, \
    monochromator_z_offset, monochromator_slit_distance, monochromator_slit_opening


# ################################################## SIM PARAMETERS ####################################################


show = False
repeats = 200

""" energy_scan(plts, bl: SKIF15): """
energies = [5.5e4]  # [3.0e4, 3.5e4, 4.0e4, 4.5e4, 5.0e4, 5.5e4, 6.0e4, 6.5e4, 7.0e4, 7.5e4]  #
dE = 1e3
de_plot_scaling = .5


# ################################################# SETUP PARAMETERS ###################################################


""" Front End Aperture """
front_end_opening = [
    -front_end_distance * np.tan(front_end_h_angle / 2.),
    front_end_distance * np.tan(front_end_h_angle / 2.),
    -front_end_distance * np.tan(front_end_v_angle / 2.),
    front_end_distance * np.tan(front_end_v_angle / 2.)
]

""" Monochromator """
monochromator_alignment_energy = 75e3
monochromator_c1_alpha = np.radians(15.)
monochromator_c1_thickness = 2.1
monochromator_c2_alpha = np.radians(15.)
monochromator_c2_thickness = 2.1

""" Sample Aperture """
sample_ap_distance = monochromator_distance + 10000  # from source
sample_ap_opening = [-20., 20., -5., 5.]


# #################################################### MATERIALS #######################################################


cr_si_1 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', t=monochromator_c1_thickness)
cr_si_2 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', t=monochromator_c2_thickness)


# #################################################### BEAMLINE ########################################################


class SKIF15(raycing.BeamLine):
    def __init__(self):
        raycing.BeamLine.__init__(self)

        self.alignE = monochromator_alignment_energy

        self.SuperCWiggler = rsources.Wiggler(
            name=r"Superconducting Wiggler",
            bl=self,
            center=[0, 0, 0],
            eMin=100,
            eMax=100000,
            **ring_kwargs,
            **wiggler_1_5_kwargs
        )

        self.FrontEnd = rapts.RectangularAperture(
            bl=self,
            name=r"Front End Slit",
            center=[0, front_end_distance, 0],
            opening=front_end_opening
        )

        self.MonochromatorCr1 = roes.BentLaueCylinder(
            bl=self,
            name=r'Si[111] Crystal 1',
            center=[0., monochromator_distance, 0.],
            pitch=np.pi / 2.,
            roll=0.,
            yaw=0.,
            alpha=monochromator_c1_alpha,
            material=(cr_si_1,),
            R=5e3,
            targetOpenCL='CPU'
        )

        self.MonochromatorCr2 = roes.BentLaueCylinder(
            bl=self,
            name=r'Si[111] Crystal 2',
            center=[0., monochromator_distance, monochromator_z_offset],
            positionRoll=np.pi,
            pitch=0.,
            roll=0.,
            yaw=0.,
            alpha=monochromator_c2_alpha,
            material=(cr_si_2,),
            R=5e3,
            targetOpenCL='CPU'
        )

        self.MonochromatorSlit = rapts.RectangularAperture(
            bl=self,
            name=r"Monochromator Slit",
            center=[0, monochromator_slit_distance, monochromator_z_offset],
            opening=monochromator_slit_opening
        )


# ################################################# BEAM TOPOLOGY ######################################################


def run_process(bl: SKIF15):

    beam_source = bl.sources[0].shine()

    beam_ap1 = bl.FrontEnd.propagate(
        beam=beam_source
    )

    beam_mono_c1_global, beam_mono_c1_local = bl.MonochromatorCr1.reflect(
        beam=beam_source
    )

    beam_mono_c2_global, beam_mono_c2_local = bl.MonochromatorCr2.reflect(
        beam=beam_mono_c1_global
    )

    beam_ap2 = bl.MonochromatorSlit.propagate(
        beam=beam_mono_c2_global
    )

    if show:
        bl.prepare_flow()

    return {
        'BeamSourceGlobal': beam_source,
        'BeamAperture1Local': beam_ap1,
        'BeamMonoC1Local': beam_mono_c1_local,
        'BeamMonoC1Global': beam_mono_c1_global,
        'BeamMonoC2Local': beam_mono_c2_local,
        'BeamMonoC2Global': beam_mono_c2_global,
        'BeamAperture2Local': beam_ap2,
    }


rrun.run_process = run_process


# ##################################################### PLOTS ##########################################################


plots = [
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title='Front End Spot',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamAperture1Local',
    #     title='Front End Directions',
    #     xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),

    xrtplot.XYCPlot(
        beam='BeamAperture2Local',
        title='DCD Slit Spot',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamAperture2Local',
    #     title='DCD Slit Directions',
    #     xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),
]


# ############################################### SEQUENCE GENERATOR ###################################################


def align_energy(bl: SKIF15, en, d_en):
    # changing energy for the beamline / source
    bl.alignE = en
    bl.SuperCWiggler.eMin = en - d_en
    bl.SuperCWiggler.eMax = en + d_en

    # Diffraction angle for the DCM
    theta0 = np.arcsin(rm.ch / (2 * bl.MonochromatorCr1.material[0].d * en))

    # Setting up DCM orientations / positions
    # Crystal 1
    bl.MonochromatorCr1.pitch = np.pi / 2 + theta0 + monochromator_c1_alpha
    bl.MonochromatorCr1.set_alpha(monochromator_c1_alpha)
    bl.MonochromatorCr1.center = [
        0.,
        monochromator_distance - monochromator_z_offset / np.tan(2. * theta0),
        0.
    ]

    # Crystal 2
    bl.MonochromatorCr2.pitch = np.pi / 2 - theta0 - monochromator_c2_alpha
    bl.MonochromatorCr2.set_alpha(-monochromator_c2_alpha)


def energy_scan(plts, bl: SKIF15):

    for ii, energy in enumerate(energies):

        align_energy(bl, energy, dE)

        # adjusting plots
        for plot in plts:
            plot.saveName = '%s-%dkeV-%s-%s.png' % (
                plot.title, int(energy * 1e-3),
                ('%02dm' % int(bl.MonochromatorCr1.R * 1e-3)) if bl.MonochromatorCr1.R < np.inf else 'inf',
                ('%02dm' % int(bl.MonochromatorCr2.R * 1e-3)) if bl.MonochromatorCr2.R < np.inf else 'inf'
            )
            plot.caxis.offset = energy
            plot.caxis.limits = [energy - de_plot_scaling * dE, energy + de_plot_scaling * dE]

        yield


# ###################################################### MAIN ##########################################################


if __name__ == '__main__':
    beamline = SKIF15()
    scan = energy_scan

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
