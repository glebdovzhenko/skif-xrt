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

from components import BentLaueCylinder, CrystalSiPrecalc
from params.sources import ring_kwargs, wiggler_1_5_kwargs
from params.params_1_5 import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle, \
    monochromator_distance, monochromator_z_offset, monochromator_x_lim, monochromator_y_lim, \
    exit_slit_distance, exit_slit_opening


# ################################################## SIM PARAMETERS ####################################################


show = False
repeats = 10

""" energy_scan(plts, bl: SKIF15): """
# scan = 'energy_scan'
# subdir = 'img/infR-escan'
# energies = [3.0e4, 5.0e4, 7.0e4, 9.0e4, 1.1e5, 1.3e5, 1.5e5]  # [7.0e4]  #
# de_over_e = 0.02
# mono = False
# de_plot_scaling = .5
# xzpr_plot_scaling = 2.

""" r1r2map """
scan = 'energy_scan'
subdir = 'img/precalc'
energies = [29e3]
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
monochromator_c1_alpha = np.radians(20.)
monochromator_c1_thickness = 2.

# #################################################### MATERIALS #######################################################


# cr_si_1 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c1_thickness)
cr_si_1 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c1_thickness)


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
        self.MonochromatorCr1.ucl = mcl.XRT_CL(r'materials.cl', targetOpenCL='CPU')

        self.Cr1Monitor = rscreens.Screen(
            bl=self,
            name=r"Crystal 1-2 Monitor",
            center=[0, monochromator_distance, .5 * monochromator_z_offset],
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

    if show:
        bl.prepare_flow()

    return {
        'BeamSourceGlobal': beam_source,
        'BeamAperture1Local': beam_ap1,
        'BeamMonoC1Local': beam_mono_c1_local,
        'BeamMonoC1Global': beam_mono_c1_global,
        'BeamMonitor1Local': beam_mon1
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
    xrtplot.XYCPlot(
        beam='BeamMonoC1Local',
        title='C1-XprZpr',
        xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonoC1Local',
        title='C1-ZZpr',
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
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
]


# ############################################### SEQUENCE GENERATOR ###################################################


def print_positions(bl: SKIF15):
    print('#' * 20, bl.name, '#' * 20)

    for element in (bl.SuperCWiggler, bl.FrontEnd,
                    bl.MonochromatorCr1, bl.Cr1Monitor):
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

        bl.MonochromatorCr1.R = 4000.

        set_de_over_e(bl.MonochromatorCr1.R)
        align_energy(bl, energy, de_over_e)

        for plot in plts:
            plot.saveName = '%s/%s-%dkeV-%.01fmm-%.01fdeg-%sm.png' % (
                subdir, plot.title,
                int(energy * 1e-3),
                monochromator_c1_thickness,
                np.degrees(monochromator_c1_alpha),
                bl.MonochromatorCr1.pretty_R(),
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
