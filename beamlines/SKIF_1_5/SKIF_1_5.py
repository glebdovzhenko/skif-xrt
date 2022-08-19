import numpy as np
from copy import deepcopy

import matplotlib
matplotlib.use('agg')

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

from components import BentLaueCylinder
from params.sources import ring_kwargs, wiggler_1_5_kwargs
from params.params_1_5 import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle, \
    monochromator_distance, monochromator_z_offset, monochromator_x_lim, monochromator_y_lim, \
    exit_slit_distance, exit_slit_opening
from utils.calc import chukhovskii_krisch_r2, max_reflectivity_t


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
scan = 'r1r2_map'
subdir = 'img/r1r2_map'
energies = np.arange(30., 155., 5.) * 1e3
de_over_e = 0.05  # 0.015
# de_over_e = 0.003     # ###########################################
mono = False
de_plot_scaling = 1.
xpr_plot_scaling = 1.2
zpr_plot_scaling = 1.5  # 10.
x_plot_scaling = 1.4
z_plot_scaling = 2.4


# ################################################# SETUP PARAMETERS ###################################################


""" Monochromator """
monochromator_alignment_energy = energies[0]
monochromator_c1_alpha = np.radians(21.9)
monochromator_c1_thickness = 2.  # max_reflectivity_t(energies[0], monochromator_c1_alpha, 1)
monochromator_c2_alpha = np.radians(21.9)
monochromator_c2_thickness = 2.  # max_reflectivity_t(energies[0], monochromator_c1_alpha, 1)


# #################################################### MATERIALS #######################################################


cr_si_1 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c1_thickness)
cr_si_2 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c2_thickness)


# #################################################### BEAMLINE ########################################################


class SKIF15(raycing.BeamLine):
    def __init__(self):
        raycing.BeamLine.__init__(self)

        self.name = r"SKIF 1-5"

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
            targetOpenCL='CPU',
            limPhysY=monochromator_y_lim,
            limOptY=monochromator_y_lim,
            limPhysX=monochromator_x_lim,
            limOptX=monochromator_x_lim,
        )
        self.MonochromatorCr1.ucl=mcl.XRT_CL(r'materials.cl', targetOpenCL='CPU')

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
            targetOpenCL='CPU',
            limPhysY=monochromator_y_lim,
            limOptY=monochromator_y_lim,
            limPhysX=monochromator_x_lim,
            limOptX=monochromator_x_lim,
        )
        self.MonochromatorCr2.ucl = mcl.XRT_CL(r'materials.cl', targetOpenCL='CPU')

        self.ExitMonitor = rscreens.Screen(
            bl=self,
            name=r"Exit Monitor",
            center=[0, exit_slit_distance - 10, monochromator_z_offset],
        )

        self.ExitSlit = rapts.RectangularAperture(
            bl=self,
            name=r"Exit Slit",
            center=[0, exit_slit_distance, monochromator_z_offset],
            opening=exit_slit_opening
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

    beam_mon1 = bl.Cr1Monitor.expose(
        beam=beam_mono_c1_global
    )

    beam_mono_c2_global, beam_mono_c2_local = bl.MonochromatorCr2.reflect(
        beam=beam_mono_c1_global
    )

    beam_mon2 = bl.ExitMonitor.expose(
        beam=beam_mono_c2_global
    )

    beam_ap2 = bl.ExitSlit.propagate(
        beam=beam_mono_c2_global
    )

    if show:
        bl.prepare_flow()

    return {
        'BeamSourceGlobal': beam_source,
        'BeamAperture1Local': beam_ap1,
        'BeamMonoC1Local': beam_mono_c1_local,
        'BeamMonoC1Global': beam_mono_c1_global,
        'BeamMonitor1Local': beam_mon1,
        'BeamMonoC2Local': beam_mono_c2_local,
        'BeamMonoC2Global': beam_mono_c2_global,
        'BeamAperture2Local': beam_ap2,
        'BeamMonitor2Local': beam_mon2,
    }


rrun.run_process = run_process


# ##################################################### PLOTS ##########################################################


plots = [
    # Front-end
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title='FE-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamAperture1Local',
    #     title="FE-XprZpr",
    #     xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamAperture1Local',
    #     title="FE-ZZpr",
    #     xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamAperture1Local',
    #     title="FE-ZE",
    #     xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
    #     yaxis=xrtplot.XYCAxis(label=r'E', unit='eV', data=raycing.get_energy),
    #     aspect='auto'),

    # C1
    xrtplot.XYCPlot(
        beam='BeamMonoC1Local',
        title='C1-XY',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$y$', unit='mm', data=raycing.get_y),
        aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamMonoC1Local',
    #     title='C1-XprZpr',
    #     xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamMonoC1Local',
    #     title='C1-ZZpr',
    #     xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonoC1Local',
        title="C1-YE",
        xaxis=xrtplot.XYCAxis(label=r'$y$', unit='mm', data=raycing.get_y),
        yaxis=xrtplot.XYCAxis(label=r'E', unit='eV', data=raycing.get_energy),
        aspect='auto'),

    # Monitor between crystals
    xrtplot.XYCPlot(
        beam='BeamMonitor1Local',
        title='C1C2-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor1Local',
        title='C1C2-XprZpr',
        xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor1Local',
        title='C1C2-ZZpr',
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor1Local',
        title="C1C2-ZE",
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'E', unit='eV', data=raycing.get_energy),
        aspect='auto'),

    # C2
    xrtplot.XYCPlot(
        beam='BeamMonoC2Local',
        title='C2-XY',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$y$', unit='mm', data=raycing.get_y),
        aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamMonoC2Local',
    #     title='C2-XprZpr',
    #     xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),
    # xrtplot.XYCPlot(
    #     beam='BeamMonoC2Local',
    #     title='C2-ZZpr',
    #     xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
    #     aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonoC2Local',
        title="C2-YE",
        xaxis=xrtplot.XYCAxis(label=r'$y$', unit='mm', data=raycing.get_y),
        yaxis=xrtplot.XYCAxis(label=r'E', unit='eV', data=raycing.get_energy),
        aspect='auto'),

    # Exit monitor
    xrtplot.XYCPlot(
        beam='BeamMonitor2Local',
        title='EM-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor2Local',
        title='EM-XprZpr',
        xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor2Local',
        title='EM-ZZpr',
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor2Local',
        title="EM-ZE",
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'E', unit='eV', data=raycing.get_energy),
        aspect='auto'),

    # Exit slit
    xrtplot.XYCPlot(
        beam='BeamAperture2Local',
        title='ES-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture2Local',
        title='ES-XprZpr',
        xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture2Local',
        title='ES-ZZpr',
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture2Local',
        title="ES-ZE",
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'E', unit='eV', data=raycing.get_energy),
        aspect='auto'),
]


# ############################################### SEQUENCE GENERATOR ###################################################


def print_positions(bl: SKIF15):
    print('#' * 20, bl.name, '#' * 20)

    for element in (bl.SuperCWiggler, bl.FrontEnd,
                    bl.MonochromatorCr1, bl.Cr1Monitor, bl.MonochromatorCr2,
                    bl.ExitMonitor, bl.ExitSlit):
        print('#' * 5, element.name, 'at', element.center)

    print('#' * (42 + len(bl.name)))


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

    print_positions(bl)


def upd_plots(plts, bl: SKIF15, en, d_en):

    # 'FE-XZ', "FE-XprZpr", "FE-ZZpr", "FE-ZE",
    # 'C1-XZ', 'C1-XprZpr', 'C1-ZZpr', "C1-ZE",
    # 'C1C2-XZ', 'C1C2-XprZpr', 'C1C2-ZZpr', "C1C2-ZE",
    # 'C2-XZ', 'C2-XprZpr', 'C2-ZZpr', "C2-ZE",
    # 'EM-XZ', 'EM-XprZpr', 'EM-ZZpr', "EM-ZE",
    # 'ES-XZ', 'ES-XprZpr', 'ES-ZZpr', "ES-ZE",

    global xpr_plot_scaling, zpr_plot_scaling, x_plot_scaling, z_plot_scaling

    for plot in plts:
        # adjusting energy limits and offset
        plot.caxis.offset = en
        plot.caxis.limits = [en * (1. - de_plot_scaling * d_en),
                             en * (1. + de_plot_scaling * d_en)]

        # By default no plots need offset
        plot.xaxis.offset = 0.
        plot.yaxis.offset = 0.

        # Except for C1C2 y-axes which is z'. They have to be offset by 1st crystal scattering direction.
        if plot.title in ('C1-XprZpr', 'C1-ZZpr', 'C1C2-XprZpr', 'C1C2-ZZpr'):
            plot.yaxis.offset = bl.Cr1Monitor.center[2] / (bl.Cr1Monitor.center[1] - bl.MonochromatorCr1.center[1])

        # "Directions" plot x-axis is x', setting appropriate limits:
        if '-XprZpr' in plot.title:
            plot.xaxis.limits = [plot.xaxis.offset + xpr_plot_scaling * front_end_opening[0] / front_end_distance,
                                 plot.xaxis.offset + xpr_plot_scaling * front_end_opening[1] / front_end_distance]

        # "Directions" and "Corr" plots y-axis is z', setting appropriate limits:
        if ('-XprZpr' in plot.title) or ('-ZZpr' in plot.title):
            if ('ES-' in plot.title) or ('EM-' in plot.title):
                plot.yaxis.limits = [plot.yaxis.offset + 1.8 * front_end_opening[2] / front_end_distance,
                                     plot.yaxis.offset + 1.8 * front_end_opening[3] / front_end_distance]
            else:
                plot.yaxis.limits = [plot.yaxis.offset + zpr_plot_scaling * front_end_opening[2] / front_end_distance,
                                     plot.yaxis.offset + zpr_plot_scaling * front_end_opening[3] / front_end_distance]

        # "Corr" plot x-axis is z, setting appropriate limits:
        if '-ZZpr' in plot.title:
            if ('C1-' in plot.title) or ('C2-' in plot.title):
                plot.xaxis.limits = [.5 * z_plot_scaling * exit_slit_opening[2],
                                     .5 * z_plot_scaling * exit_slit_opening[3]]
            else:
                plot.xaxis.limits = [z_plot_scaling * exit_slit_opening[2],
                                     z_plot_scaling * exit_slit_opening[3]]

        # For "Spot" plots setting the scale to the exit slit size times scaling
        if ('-XZ' in plot.title) or ('-XY' in plot.title):
            if ('C1-' in plot.title) or ('C2-' in plot.title):
                plot.xaxis.limits = [.5 * x_plot_scaling * exit_slit_opening[0],
                                     .5 * x_plot_scaling * exit_slit_opening[1]]
                plot.yaxis.limits = [.5 * z_plot_scaling * exit_slit_opening[2],
                                     .5 * z_plot_scaling * exit_slit_opening[3]]
            else:
                plot.xaxis.limits = [x_plot_scaling * exit_slit_opening[0],
                                     x_plot_scaling * exit_slit_opening[1]]
                plot.yaxis.limits = [z_plot_scaling * exit_slit_opening[2],
                                     z_plot_scaling * exit_slit_opening[3]]

        if ('-ZE' in plot.title) or ('-YE' in plot.title):
            if ('C1-' in plot.title) or ('C2-' in plot.title):
                plot.xaxis.limits = [.5 * z_plot_scaling * exit_slit_opening[2],
                                     .5 * z_plot_scaling * exit_slit_opening[3]]
            else:
                plot.xaxis.limits = [z_plot_scaling * exit_slit_opening[2],
                                     z_plot_scaling * exit_slit_opening[3]]
            plot.yaxis.offset = en
            plot.yaxis.limits = [en * (1. - de_plot_scaling * d_en),
                                 en * (1. + de_plot_scaling * d_en)]


def set_de_over_e(radius):
    global de_over_e, z_plot_scaling, zpr_plot_scaling

    if radius < 4.e3:
        de_over_e = 1.e4 / 70.e3
        zpr_plot_scaling = 100.
    elif 4.e3 <= radius < 8.e3:
        de_over_e = 3.e3 / 70.e3
        zpr_plot_scaling = 25.
    elif 8.e3 <= radius < 15.e3:
        de_over_e = 1.e3 / 70.e3
        zpr_plot_scaling = 10.
    elif 15.e3 <= radius < 21.e3:
        de_over_e = 5.e2 / 70.e3
        zpr_plot_scaling = 5.
    elif 21.e3 <= radius < 28.e3:
        de_over_e = 2.e2 / 70.e3
        zpr_plot_scaling = 3.
    elif 28.e3 <= radius < 30.e3:
        de_over_e = 1.e2 / 70.e3
        zpr_plot_scaling = 2.
    elif 30.e3 <= radius < 39.e3:
        de_over_e = 5.e1 / 70.e3
        zpr_plot_scaling = 2.
    elif 39.e3 <= radius < 52.e3:
        de_over_e = 1.e2 / 70.e3
        zpr_plot_scaling = 2.
    elif 52.e3 <= radius < 100.e3:
        de_over_e = 2.e2 / 70.e3
        zpr_plot_scaling = 2.
    elif 100.e3 <= radius:
        de_over_e = 4.e2 / 70.e3
        zpr_plot_scaling = 2.

    if radius < 3.e3:
        z_plot_scaling = 2.4
    if 3.e3 <= radius:
        z_plot_scaling = 2.4

    de_over_e *= 1.5


def energy_scan(plts, bl: SKIF15):

    for ii, energy in enumerate(energies):

        bl.MonochromatorCr1.R = 1e3 * monochromator_c1_thickness
        bl.MonochromatorCr2.R = np.inf
        set_de_over_e(bl.MonochromatorCr1.R)

        align_energy(bl, energy, de_over_e)

        for plot in plts:
            plot.saveName = '%s/%s-%dkeV-%.01fdeg-%sm-%sm.png' % (
                subdir, plot.title,
                int(energy * 1e-3),
                np.degrees(monochromator_c1_alpha),
                bl.MonochromatorCr1.pretty_R(),
                bl.MonochromatorCr2.pretty_R()
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        upd_plots(plts, bl, energy, de_over_e)
        yield


def r1_scan(plts, bl: SKIF15):
    global de_over_e
    energy = energies[0]

    align_energy(bl, energy, de_over_e)
    bl.MonochromatorCr1.R = np.inf
    bl.MonochromatorCr2.R = np.inf

    r1s = np.concatenate((np.arange(1., 20.1, .5), [np.inf])) * 1e3
    # r1s = (np.arange(.7, 5.1, .05)) * 1e3
    # r1s = np.array([np.inf])

    for r1 in r1s:
        bl.MonochromatorCr1.R = r1
        bl.MonochromatorCr2.R = np.inf

        set_de_over_e(r1)
        align_energy(bl, energy, de_over_e)

        for plot in plts:
            plot.saveName = '%s/%s-%dkeV-%sm-%sm.png' % (
                subdir, plot.title,
                int(energy * 1e-3),
                bl.MonochromatorCr1.pretty_R(),
                bl.MonochromatorCr2.pretty_R(),
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        upd_plots(plts, bl, energy, de_over_e)

        yield


def r1r2_map(plts, bl: SKIF15):
    global de_over_e
    energy = energies[0]

    align_energy(bl, energy, de_over_e)
    bl.MonochromatorCr1.R = np.inf
    bl.MonochromatorCr2.R = np.inf

    r1s = np.array([30.]) * 1e3
    # r2s = chukhovskii_krisch_r2(r1s,
    #                             theta=np.arcsin(rm.ch / (2 * bl.MonochromatorCr1.material[0].d * energy)),
    #                             c1chi=monochromator_c1_alpha,
    #                             c2chi=monochromator_c2_alpha,
    #                             source_c1_y=monochromator_distance,
    #                             c1_c2_z=monochromator_z_offset)
    r2s = r1s + 300

    for r1, r2 in zip(r1s, r2s):
        for r2 in np.arange(r2-1000, r2 + 1000, 10):
            bl.MonochromatorCr1.R = r1
            bl.MonochromatorCr2.R = r2

            set_de_over_e(r1)
            align_energy(bl, energy, de_over_e)

            for plot in plts:
                plot.saveName = '%s/%s-%dkeV-%.01fdeg-%sm-%sm.png' % (
                    subdir, plot.title,
                    int(energy * 1e-3),
                    np.degrees(monochromator_c1_alpha),
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

    # r1s = np.arange(1., 12., 1.) * 1e3
    # r2s = chukhovskii_krisch_r2(r1s,
    #                             theta=np.arcsin(rm.ch / (2 * bl.MonochromatorCr1.material[0].d * energy)),
    #                             c1chi=monochromator_c1_alpha,
    #                             c2chi=monochromator_c2_alpha,
    #                             source_c1_y=monochromator_distance,
    #                             c1_c2_z=monochromator_z_offset)

    r1s, r2s = np.array([np.inf]), np.array([np.inf])

    for r1, r2 in zip(r1s, r2s):
        bl.MonochromatorCr1.R = r1
        bl.MonochromatorCr2.R = r2

        set_de_over_e(r1)
        align_energy(bl, energy, de_over_e)

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


def alpha_scan(plts, bl: SKIF15):
    global de_over_e, monochromator_c1_alpha, monochromator_c1_thickness, monochromator_c2_alpha, monochromator_c2_thickness
    energy = energies[0]

    align_energy(bl, energy, de_over_e)
    bl.MonochromatorCr1.R = np.inf
    bl.MonochromatorCr2.R = np.inf

    alphas = np.radians(np.arange(0., 30., 1.))

    for alpha in alphas:
        t = max_reflectivity_t(energy, alpha, 1)

        monochromator_c1_alpha = alpha
        monochromator_c2_alpha = alpha

        monochromator_c1_thickness = t
        monochromator_c2_thickness = t

        bl.MonochromatorCr1.material[0].t = monochromator_c1_thickness
        bl.MonochromatorCr2.material[0].t = monochromator_c2_thickness

        bl.MonochromatorCr1.set_alpha(monochromator_c1_alpha)
        bl.MonochromatorCr2.set_alpha(monochromator_c2_alpha)

        set_de_over_e(np.inf)
        align_energy(bl, energy, de_over_e)

        for plot in plts:
            plot.saveName = '%s/%s-%dkeV-%sdeg.png' % (
                subdir, plot.title,
                int(energy * 1e-3),
                np.degrees(alpha)
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        upd_plots(plts, bl, energy, de_over_e)

        yield


def t_scan(plts, bl: SKIF15):
    global de_over_e, monochromator_c1_alpha, monochromator_c1_thickness, monochromator_c2_alpha, monochromator_c2_thickness
    energy = energies[0]

    align_energy(bl, energy, de_over_e)
    bl.MonochromatorCr1.R = np.inf
    bl.MonochromatorCr2.R = np.inf

    ts = [max_reflectivity_t(energy, monochromator_c1_alpha, 2), max_reflectivity_t(energy, monochromator_c1_alpha, 4)]
    ts = np.linspace(*ts, 60)

    for t, in ts:
        monochromator_c1_thickness = t
        monochromator_c2_thickness = t

        bl.MonochromatorCr1.material[0].t = monochromator_c1_thickness
        bl.MonochromatorCr2.material[0].t = monochromator_c2_thickness

        set_de_over_e(np.inf)
        align_energy(bl, energy, de_over_e)

        for plot in plts:
            plot.saveName = '%s/%s-%dkeV-%.04fmm.png' % (
                subdir, plot.title,
                int(energy * 1e-3),
                t
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
