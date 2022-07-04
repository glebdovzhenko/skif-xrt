import numpy as np

import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from components import BentLaueCylinder
from params.sources import ring_kwargs, wiggler_1_5_kwargs
from params.params_1_5 import front_end_distance, front_end_opening, front_end_h_angle, front_end_v_angle, \
    monochromator_distance


# ################################################## SIM PARAMETERS ####################################################


show = False
repeats = 100

# """ scan1 """
# scan = 'energy_scan'
# subdir = 'img/'
# energies = [7.0e4]
# d_energy = 1.0e3
# ap_size = None
# de_plot_scaling = .2
# xzpr_plot_scaling = 3.5
# xz_plot_scaling = 1.1

# """ scan2 """
# scan = 'alpha_scan'
# subdir = 'img/70keV-Alpscan'
# energies = [7.0e4]
# d_energy = 1.0e3
# ap_size = None
# de_plot_scaling = .2
# xzpr_plot_scaling = 3.5
# xz_plot_scaling = 1.1

""" scan3 """
# scan = 't_alp_maxs'
# subdir = 'img/70keV-t-alp-maxs'
# energies = [7.0e4]
# d_energy = 1.0e3
# ap_size = None
# de_plot_scaling = .2
# xzpr_plot_scaling = 3.5
# xz_plot_scaling = 1.1

""" scan4 """
scan = 'r_scan'
subdir = 'img/70keV-Rscan'
energies = [7.0e4]
d_energy = 1.5e3
ap_size = [-50., 50., -30, 30]
de_plot_scaling = .5
xzpr_plot_scaling = 12.5
xz_plot_scaling = 1.1


# ################################################# SETUP PARAMETERS ###################################################


""" Monochromator """
monochromator_c1_alpha = np.radians(1.)
monochromator_c1_thickness = 0.73
monochromator_c1_radius = np.inf


""" Sample Aperture """
sample_ap_distance = 2. * monochromator_distance  # from source


# #################################################### MATERIALS #######################################################


si111 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', t=monochromator_c1_thickness)


# #################################################### BEAMLINE ########################################################


class RotatingArmBL(raycing.BeamLine):
    def __init__(self):
        raycing.BeamLine.__init__(self)

        self.alignE = 75e3

        self.SuperCWiggler = rsources.Wiggler(
            name=r"Superconducting Wiggler",
            bl=self,
            center=[0, 0, 0],
            eMin=100,
            eMax=150000,
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
            material=(si111,),
            R=monochromator_c1_radius,
            targetOpenCL='CPU'
        )

        self.SampleAperture = rapts.RectangularAperture(
            bl=self,
            name=r"Sample Aperture",
            center=[0, sample_ap_distance, 0],
        )


# ################################################# BEAM TOPOLOGY ######################################################


def run_process(bl: RotatingArmBL):

    beam_source = bl.sources[0].shine()

    beam_ap1 = bl.FrontEnd.propagate(
        beam=beam_source
    )
    beam_mono_c1_global, beam_mono_c1_local = bl.MonochromatorCr1.reflect(
        beam=beam_source
    )
    beam_ap2 = bl.SampleAperture.propagate(
        beam=beam_mono_c1_global
    )

    if show:
        bl.prepare_flow()

    return {
        'BeamSourceGlobal': beam_source,
        'BeamAperture1Local': beam_ap1,
        'BeamMonoC1Local': beam_mono_c1_local,
        'BeamMonoC1Global': beam_mono_c1_global,
        'BeamAperture2Local': beam_ap2,
    }


rrun.run_process = run_process


# ##################################################### PLOTS ##########################################################


plots = [
    # xrtplot.XYCPlot(
    #     beam='BeamAperture1Local',
    #     title='Front End Spot',
    #     xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x,
    #                           limits=[xz_plot_scaling * front_end_opening[0],
    #                                   xz_plot_scaling * front_end_opening[1]]),
    #     yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z,
    #                           limits=[xz_plot_scaling * front_end_opening[2],
    #                                   xz_plot_scaling * front_end_opening[3]]),
    #     aspect='auto'),
    #
    # xrtplot.XYCPlot(
    #     beam='BeamAperture1Local',
    #     title='Front End Directions',
    #     xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime,
    #                           limits=[xzpr_plot_scaling * front_end_opening[0] / front_end_distance,
    #                                   xzpr_plot_scaling * front_end_opening[1] / front_end_distance]
    #                           ),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime,
    #                           limits=[xzpr_plot_scaling * front_end_opening[2] / front_end_distance,
    #                                   xzpr_plot_scaling * front_end_opening[3] / front_end_distance]
    #                           ),
    #     aspect='auto'),
    #
    # xrtplot.XYCPlot(
    #     beam='BeamAperture1Local',
    #     title='Front End Corr',
    #     xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z,
    #                           limits=[xz_plot_scaling * front_end_opening[2],
    #                                   xz_plot_scaling * front_end_opening[3]]),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime,
    #                           limits=[xzpr_plot_scaling * front_end_opening[2] / front_end_distance,
    #                                   xzpr_plot_scaling * front_end_opening[3] / front_end_distance]
    #                           ),
    #     aspect='auto'),

    xrtplot.XYCPlot(
        beam='BeamAperture2Local',
        title='SCM Slit Spot',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),

    xrtplot.XYCPlot(
        beam='BeamAperture2Local',
        title='SCM Slit Directions',
        xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime,
                              limits=[xzpr_plot_scaling * front_end_opening[0] / front_end_distance,
                                      xzpr_plot_scaling * front_end_opening[1] / front_end_distance]
                              ),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime,
                              limits=[xzpr_plot_scaling * front_end_opening[2] / front_end_distance,
                                      xzpr_plot_scaling * front_end_opening[3] / front_end_distance]
                              ),
        aspect='auto'),

    # xrtplot.XYCPlot(
    #     beam='BeamAperture2Local',
    #     title='SCM Slit Corr',
    #     xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
    #     yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime,
    #                           limits=[xzpr_plot_scaling * front_end_opening[2] / front_end_distance,
    #                                   xzpr_plot_scaling * front_end_opening[3] / front_end_distance]
    #                           ),
    #     aspect='auto'),
]


# ############################################### SEQUENCE GENERATOR ###################################################


def align_energy(bl: RotatingArmBL, en, d_en, ap_s):
    # changing energy for the beamline / source
    bl.alignE = en
    bl.SuperCWiggler.eMin = en - d_en
    bl.SuperCWiggler.eMax = en + d_en

    # Diffraction angle for the DCM
    theta0 = np.arcsin(rm.ch / (2 * bl.MonochromatorCr1.material[0].d * en))

    # Setting up SCM orientation
    bl.MonochromatorCr1.pitch = np.pi / 2 + theta0 + monochromator_c1_alpha
    bl.MonochromatorCr1.set_alpha(monochromator_c1_alpha)

    # Moving aperture
    bl.SampleAperture.center = [
        0.,
        monochromator_distance + (sample_ap_distance - monochromator_distance) * np.cos(2 * theta0),
        (sample_ap_distance - monochromator_distance) * np.sin(2 * theta0),
    ]

    if ap_s is not None:
        bl.SampleAperture.opening = ap_s
    else:
        bl.SampleAperture.opening = [ii * sample_ap_distance / front_end_distance for ii in front_end_opening]

    ap_s = (bl.SampleAperture.opening[3] - bl.SampleAperture.opening[2]) / 2.

    print(
        '#### Monochromator: E = %.01f keV; Θ = %.03f°; Rx = %.03f°\n'
        '#### Beamline: E = %.01f keV ± %.01f eV\n'
        '#### Aperture: Δz = ±%.03f mm' % (
            en * 1e-3, np.degrees(theta0), np.degrees(bl.MonochromatorCr1.pitch), en * 1e-3, d_en, ap_s,
        ))


def upd_plots(plts, bl: RotatingArmBL, en, d_en):
    for plot in plts:
        plot.caxis.offset = en
        plot.caxis.limits = [en - de_plot_scaling * d_en,
                             en + de_plot_scaling * d_en]

        # adjusting z' offset
        if plot.title == 'SCM Slit Directions':
            plot.yaxis.offset = bl.SampleAperture.center[2] / \
                                (bl.SampleAperture.center[1] - bl.MonochromatorCr1.center[1])
            plot.yaxis.limits = [plot.yaxis.offset + xzpr_plot_scaling * front_end_opening[2] / front_end_distance,
                                 plot.yaxis.offset + xzpr_plot_scaling * front_end_opening[3] / front_end_distance]

        # adjusting xz limits
        if plot.title == 'SCM Slit Spot':
            plot.xaxis.limits = [xz_plot_scaling * bl.SampleAperture.opening[0],
                                 xz_plot_scaling * bl.SampleAperture.opening[1]]
            plot.yaxis.limits = [xz_plot_scaling * bl.SampleAperture.opening[2],
                                 xz_plot_scaling * bl.SampleAperture.opening[3]]

        # adjusting the correlation plot
        if plot.title == 'SCM Slit Corr':
            plot.xaxis.limits = [xz_plot_scaling * bl.SampleAperture.opening[0],
                                 xz_plot_scaling * bl.SampleAperture.opening[1]]
            plot.yaxis.offset = bl.SampleAperture.center[2] / \
                                (bl.SampleAperture.center[1] - bl.MonochromatorCr1.center[1])
            plot.yaxis.limits = [plot.yaxis.offset + xzpr_plot_scaling * front_end_opening[2] / front_end_distance,
                                 plot.yaxis.offset + xzpr_plot_scaling * front_end_opening[3] / front_end_distance]


def energy_scan(plts, bl: RotatingArmBL):
    for energy in energies:
        align_energy(bl, energy, d_energy, ap_size)

        # setting plot names
        for plot in plts:
            plot.saveName = '%s/%s-%.01fkeV-.png' % (
                subdir, plot.title, energy * 1e-3
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        # adjusting plot axes
        upd_plots(plts, bl, energy, d_energy)

        yield


def r_scan(plts, bl: RotatingArmBL):
    global xzpr_plot_scaling

    energy = energies[0]
    align_energy(bl, energy, d_energy, ap_size)

    rs = np.arange(150., 2050., 50.)
    # rs = np.concatenate((rs, [np.inf]))
    # rs = np.concatenate((rs, np.arange(25., 40., 1.)))

    for r in rs:

        if r < 15:
            xzpr_plot_scaling = 10
        elif 25 > r >= 15:
            xzpr_plot_scaling = 5
        else:
            xzpr_plot_scaling = 2.5

        bl.MonochromatorCr1.R = r * 1e3

        # setting plot names
        for plot in plts:
            plot.saveName = '%s/%s-%.01fm.png' % (
                subdir, plot.title, r
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        # adjusting plot axes
        upd_plots(plts, bl, energy, d_energy)

        yield


def alpha_scan(plts, bl: RotatingArmBL):

    global monochromator_c1_alpha
    energy = energies[0]

    for alpha in np.linspace(0.1, 30.1, 60):

        monochromator_c1_alpha = np.radians(alpha)
        align_energy(bl, energy, d_energy, ap_size)

        # adjusting plots
        for plot in plts:
            plot.saveName = '%s/%s-%.01fdeg.png' % (
                subdir, plot.title, alpha
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        upd_plots(plts, bl, energy, d_energy)

        yield


def d_scan(plts, bl: RotatingArmBL):

    global monochromator_c1_thickness
    energy = energies[0]
    align_energy(bl, energy, d_energy, ap_size)

    for monochromator_c1_thickness in np.arange(1.60, 3.3, 0.05):

        bl.MonochromatorCr1.material[0].t = monochromator_c1_thickness

        # adjusting plots
        for plot in plts:
            plot.saveName = '%s/%s-%.02fmm.png' % (
                subdir, plot.title, monochromator_c1_thickness
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        upd_plots(plts, bl, energy, d_energy)

        yield


def t_alp_maxs(plts, bl: RotatingArmBL):
    global monochromator_c1_thickness, monochromator_c1_alpha
    energy = energies[0]

    for alpha in [1., 29]:
        monochromator_c1_alpha = np.radians(alpha)
        align_energy(bl, energy, d_energy, ap_size)

        if np.isclose(alpha, 1.):
            ts = [0.23, 0.40, 0.57, 0.735, 0.90, 1.07, 1.23, 1.40, 1.57, 1.73, 1.90, 2.07, 2.23, 2.40]
        elif np.isclose(alpha, 29.):
            ts = [0.2, 0.35, 0.49, 0.64, 0.79, 0.93, 1.08, 1.22, 1.37, 1.51, 1.66, 1.81, 1.95, 2.10, 2.24, 2.38]
        else:
            ts = []

        for monochromator_c1_thickness in ts:
            bl.MonochromatorCr1.material[0].t = monochromator_c1_thickness

            # adjusting plots
            for plot in plts:
                plot.saveName = '%s/%s-%.02fmm-%.01fdeg.png' % (
                    subdir, plot.title, monochromator_c1_thickness, alpha
                )
                plot.persistentName = plot.saveName.replace('.png', '.pickle')

            upd_plots(plts, bl, energy, d_energy)

            yield


# ###################################################### MAIN ##########################################################


if __name__ == '__main__':
    beamline = RotatingArmBL()
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
