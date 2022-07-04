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

from components import BentLaueCylinder
from params.sources import ring_kwargs, wiggler_1_5_kwargs
from params.params_1_5 import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle, \
    monochromator_distance, monochromator_z_offset, exit_slit_distance, exit_slit_opening


# ################################################## SIM PARAMETERS ####################################################


show = False
repeats = 1

""" energy_scan(plts, bl: SKIF15): """
# scan = 'energy_scan'
# subdir = 'img/infR-escan'
# energies = [3.0e4, 5.0e4, 7.0e4, 9.0e4, 1.1e5, 1.3e5, 1.5e5]  # [7.0e4]  #
# de_over_e = 0.02
# mono = False
# de_plot_scaling = .5
# xzpr_plot_scaling = 2.

""" r1r2map """
scan = 'r1r2_scan'
subdir = 'img/test'
energies = [7.0e4]
de_over_e = 0.01
mono = False
de_plot_scaling = 1.
xzpr_plot_scaling = 2.


# ################################################# SETUP PARAMETERS ###################################################


""" Monochromator """
monochromator_alignment_energy = 70e3
monochromator_c1_alpha = np.radians(29.)
monochromator_c1_thickness = 0.2
monochromator_c2_alpha = np.radians(29.)
monochromator_c2_thickness = 0.2
# monochromator_c1_r = 15000
# monochromator_c2_r = 15000


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
            xPrimeMax=front_end_h_angle * .505e3,
            zPrimeMax=front_end_v_angle * .505e3,
            **ring_kwargs,
            **wiggler_1_5_kwargs
        )

        self.FrontEnd = rapts.RectangularAperture(
            bl=self,
            name=r"Front End Slit",
            center=[0, front_end_distance, 0],
            opening=front_end_opening
        )

        self.MonochromatorCr1 = BentLaueCylinder(
            bl=self,
            name=r'Si[111] Crystal 1',
            center=[0., monochromator_distance, 0.],
            pitch=np.pi / 2.,
            roll=0.,
            yaw=0.,
            alpha=monochromator_c1_alpha,
            material=(cr_si_1,),
            R=np.inf,
            targetOpenCL='CPU'
        )

        self.Cr1Monitor = rscreens.Screen(
            bl=self,
            name=r"Crystal 1-2 Monitor",
            center=[0, monochromator_distance, .5 * monochromator_z_offset],
        )

        self.MonochromatorCr2 = BentLaueCylinder(
            bl=self,
            name=r'Si[111] Crystal 2',
            center=[0., monochromator_distance, monochromator_z_offset],
            positionRoll=np.pi,
            pitch=0.,
            roll=0.,
            yaw=0.,
            alpha=monochromator_c2_alpha,
            material=(cr_si_2,),
            R=np.inf,
            targetOpenCL='CPU'
        )

        # self.MonochromatorSlit = rapts.RectangularAperture(
        #     bl=self,
        #     name=r"Monochromator Slit",
        #     center=[0, monochromator_slit_distance, monochromator_z_offset],
        #     opening=monochromator_slit_opening[:2] + [-2.5, 2.5]
        # )

        self.MonochromatorMonitor = rscreens.Screen(
            bl=self,
            name=r"Monochromator Slit",
            center=[0, exit_slit_distance, monochromator_z_offset],
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

    # beam_ap2 = bl.MonochromatorSlit.propagate(
    #     beam=beam_mono_c2_global
    # )
    beam_ap2 = bl.MonochromatorMonitor.expose(
        beam=beam_mono_c2_global
    )

    beam_ap3 = bl.Cr1Monitor.expose(
        beam=beam_mono_c1_global
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
        'BeamAperture3Local': beam_ap3,
    }


rrun.run_process = run_process


# ##################################################### PLOTS ##########################################################


plots = [
    # Front-end
    # xrtplot.XYCPlot(
    #     beam='BeamAperture1Local',
    #     title='Front End Spot',
    #     xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
    #     yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
    #     aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamAperture1Local',
    #     title='Front End Directions',
    #     xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamAperture1Local',
    #     title='Front End Corr',
    #     xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),

    # Monitor between crystals
    xrtplot.XYCPlot(
        beam='BeamAperture3Local',
        title='C1C2 Monitor Spot',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture3Local',
        title='C1C2 Monitor Directions',
        xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamAperture3Local',
    #     title='C1C2 Monitor Corr',
    #     xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),

    # Exit
    xrtplot.XYCPlot(
        beam='BeamAperture2Local',
        title='DCM Slit Spot',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture2Local',
        title='DCM Slit Directions',
        xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamAperture2Local',
    #     title='DCM Slit Corr',
    #     xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),
]


# ############################################### SEQUENCE GENERATOR ###################################################


def align_energy(bl: SKIF15, en, d_en):
    # changing energy for the beamline / source
    bl.alignE = en
    if not mono:
        bl.SuperCWiggler.eMin = en * (1. - d_en)
        bl.SuperCWiggler.eMax = en * (1. + d_en)
    else:
        bl.SuperCWiggler.eMin = en - 1.
        bl.SuperCWiggler.eMax = en + 1.

    # Diffraction angle for the DCM
    theta0 = np.arcsin(rm.ch / (2 * bl.MonochromatorCr1.material[0].d * en))

    # Setting up DCM orientations / positions
    # Crystal 1
    bl.MonochromatorCr1.pitch = np.pi / 2 + theta0 + monochromator_c1_alpha
    bl.MonochromatorCr1.set_alpha(monochromator_c1_alpha)
    bl.MonochromatorCr1.center = [
        0.,
        monochromator_distance,
        0.
    ]

    # Crystal 2
    bl.MonochromatorCr2.pitch = np.pi / 2 - theta0 - monochromator_c2_alpha
    bl.MonochromatorCr2.set_alpha(-monochromator_c2_alpha)
    bl.MonochromatorCr2.center = [
        0.,
        monochromator_distance + monochromator_z_offset / np.tan(2. * theta0),
        monochromator_z_offset
    ]

    # between-crystals monitor
    bl.Cr1Monitor.center = [
        0.,
        monochromator_distance + .5 * monochromator_z_offset / np.tan(2. * theta0),
        .5 * monochromator_z_offset
    ]


def upd_plots(plts, bl: SKIF15, en, d_en):
    for plot in plts:
        plot.caxis.offset = en
        plot.caxis.limits = [en * (1. - de_plot_scaling * d_en),
                             en * (1. + de_plot_scaling * d_en)]

        if plot.title in ('DCM Slit Directions', 'C1C2 Monitor Directions'):
            plot.yaxis.offset = 0.
            plot.yaxis.limits = [plot.yaxis.offset + xzpr_plot_scaling * front_end_opening[2] / front_end_distance,
                                     plot.yaxis.offset + xzpr_plot_scaling * front_end_opening[3] / front_end_distance]
            plot.xaxis.offset = 0.
            plot.xaxis.limits = [plot.xaxis.offset + xzpr_plot_scaling * front_end_opening[0] / front_end_distance,
                                 plot.xaxis.offset + xzpr_plot_scaling * front_end_opening[1] / front_end_distance]

        if plot.title == ('DCM Slit Corr', 'C1C2 Monitor Corr'):
            plot.yaxis.offset = 0.
            plot.yaxis.limits = [plot.yaxis.offset + xzpr_plot_scaling * front_end_opening[2] / front_end_distance,
                                 plot.yaxis.offset + xzpr_plot_scaling * front_end_opening[3] / front_end_distance]

        # # adjusting xz limits
        # if plot.title == 'SCM Slit Spot':
        #     plot.xaxis.limits = [xz_plot_scaling * bl.SampleAperture.opening[0],
        #                          xz_plot_scaling * bl.SampleAperture.opening[1]]
        #     plot.yaxis.limits = [xz_plot_scaling * bl.SampleAperture.opening[2],
        #                          xz_plot_scaling * bl.SampleAperture.opening[3]]


def energy_scan(plts, bl: SKIF15):

    for ii, energy in enumerate(energies):

        align_energy(bl, energy, de_over_e)

        bl.MonochromatorCr1.R = np.inf
        bl.MonochromatorCr2.R = np.inf

        for plot in plts:
            plot.saveName = '%s/%s-%dkeV-%sm-%sm.png' % (
                subdir, plot.title,
                int(energy * 1e-3),
                bl.MonochromatorCr1.pretty_R(),
                bl.MonochromatorCr2.pretty_R()
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        upd_plots(plts, bl, energy, de_over_e)
        yield


def r1r2_map(plts, bl: SKIF15):
    energy = energies[0]

    align_energy(bl, energy, de_over_e)
    bl.MonochromatorCr1.R = np.inf
    bl.MonochromatorCr2.R = np.inf

    r1s = np.arange(10., 100., 5.)
    r2s = np.arange(10., 100., 5.)

    for r1 in r1s:
        for r2 in r2s:
            bl.MonochromatorCr1.R = r1 * 1e3
            bl.MonochromatorCr2.R = r2 * 1e3

            for plot in plts:
                plot.saveName = '%s/%s-%dkeV-%sm-%sm.png' % (
                    subdir, plot.title,
                    int(energy * 1e-3),
                    bl.MonochromatorCr1.pretty_R(),
                    bl.MonochromatorCr2.pretty_R()
                )
                plot.persistentName = plot.saveName.replace('.png', '.pickle')

            upd_plots(plts, bl, energy, de_over_e)

            yield


def r1r2_scan(plts, bl: SKIF15):
    energy = energies[0]

    align_energy(bl, energy, de_over_e)
    bl.MonochromatorCr1.R = np.inf
    bl.MonochromatorCr2.R = np.inf

    rs = np.arange(10., 30., 1.)

    for r in rs:
        bl.MonochromatorCr1.R = r * 1e3
        bl.MonochromatorCr2.R = r * 1e3

        for plot in plts:
            plot.saveName = '%s/%s-%dkeV-%sm-%sm.png' % (
                subdir, plot.title,
                int(energy * 1e-3),
                bl.MonochromatorCr1.pretty_R(),
                bl.MonochromatorCr2.pretty_R()
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        upd_plots(plts, bl, energy, de_over_e)

        yield


# ###################################################### MAIN ##########################################################


if __name__ == '__main__':
    beamline = SKIF15()
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
