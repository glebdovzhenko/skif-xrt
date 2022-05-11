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
from params.params_1_5 import front_end_distance, front_end_h_angle, front_end_v_angle, monochromator_distance


# ################################################## SIM PARAMETERS ####################################################


show = False
repeats = 100

""" energy_scan(plts, bl: SKIF15): """
energies = [5.0e4, 5.5e4, 6.0e4, 6.5e4, 7.0e4, 7.5e4]
# between d_energies, ap_sizes if one is 'auto', it will be calculated from the other one.
d_energies = [500.] * len(energies)
ap_sizes = [2.] * len(energies)


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
monochromator_alpha = 0
monochromator_thickness = 2.1

""" Sample Aperture """
sample_ap_distance = monochromator_distance + 1000  # from source
sample_ap_opening = [-1, 1, -1, 1]


# #################################################### MATERIALS #######################################################


si111 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', t=monochromator_thickness)


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
            eMax=20000,
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
            center=['auto', monochromator_distance, 'auto'],
            pitch=0.,
            roll=0.,
            yaw=0.,
            alpha=monochromator_alpha,
            material=(si111, ),
            targetOpenCL='CPU'
        )

        self.SampleAperture = rapts.RectangularAperture(
            bl=self,
            name=r"Sample Aperture",
            center=[0, sample_ap_distance, 0],
            opening=sample_ap_opening
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


x_kwds = {
    'label': r'$x$',
    'unit': 'mm',
    'data': raycing.get_x,
    'limits': (1.4 * front_end_opening[0], 1.4 * front_end_opening[1])
}
z_kwds = {
    'label': r'$z$',
    'unit': 'mm',
    'data': raycing.get_z,
    'limits': (1.4 * front_end_opening[2], 1.4 * front_end_opening[3])
}
xpr_kwds = {
    'label': r'$x^{\prime}$',
    'unit': '',
    'data': raycing.get_xprime
}
zpr_kwds = {
    'label': r'$z^{\prime}$',
    'unit': '',
    'data': raycing.get_zprime
}

plots = [
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title='Front End Spot',
        xaxis=xrtplot.XYCAxis(**x_kwds),
        yaxis=xrtplot.XYCAxis(**z_kwds),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title='Front End Directions',
        xaxis=xrtplot.XYCAxis(**xpr_kwds),
        yaxis=xrtplot.XYCAxis(**zpr_kwds),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture2Local',
        title='Aperture X-section',
        xaxis=xrtplot.XYCAxis(**x_kwds),
        yaxis=xrtplot.XYCAxis(**z_kwds),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture2Local',
        title='Aperture Directions',
        xaxis=xrtplot.XYCAxis(**xpr_kwds),
        yaxis=xrtplot.XYCAxis(**zpr_kwds),
        aspect='auto'),
]


# ############################################### SEQUENCE GENERATOR ###################################################


def energy_scan(plts, bl: SKIF15):
    global d_energies, ap_sizes

    if d_energies == 'auto' and ap_sizes != 'auto':
        d_energies = [None] * len(ap_sizes)
    elif ap_sizes == 'auto' and d_energies != 'auto':
        ap_sizes = [None] * len(d_energies)

    for ii, (energy, d_energy, ap_size) in enumerate(zip(energies, d_energies, ap_sizes)):

        theta0 = np.arcsin(rm.ch / (2 * bl.MonochromatorCr1.material[0].d * energy))
        pitch = np.pi / 2 + theta0 + monochromator_alpha

        if ap_size is None:
            theta_min = np.arcsin(rm.ch / (2 * bl.MonochromatorCr1.material[0].d * (energy + d_energy)))
            theta_max = np.arcsin(rm.ch / (2 * bl.MonochromatorCr1.material[0].d * (energy - d_energy)))
            slit_min = (sample_ap_distance - monochromator_distance) * np.sin(2 * theta_min)
            slit_max = (sample_ap_distance - monochromator_distance) * np.sin(2 * theta_max)
            slit_c = (sample_ap_distance - monochromator_distance) * np.sin(2 * theta0)
            ap_size = max([slit_c - slit_min, slit_max - slit_c])
        elif d_energy is None:
            slit_c = (sample_ap_distance - monochromator_distance) * np.sin(2 * theta0)
            theta_min = np.arcsin((slit_c - ap_size) / (sample_ap_distance - monochromator_distance)) / 2.
            theta_max = np.arcsin((slit_c + ap_size) / (sample_ap_distance - monochromator_distance)) / 2.
            energy_max = rm.ch / (2 * bl.MonochromatorCr1.material[0].d * np.sin(theta_min))
            energy_min = rm.ch / (2 * bl.MonochromatorCr1.material[0].d * np.sin(theta_max))
            d_energy = max([energy - energy_min, energy_max - energy])

        # changing energy for the beamline / source
        bl.alignE = energy
        bl.SuperCWiggler.eMin = energy - 2. * d_energy
        bl.SuperCWiggler.eMax = energy + 2. * d_energy

        # changing monochromator orientation
        bl.MonochromatorCr1.pitch = pitch
        bl.MonochromatorCr1.set_alpha(monochromator_alpha)

        # changing aperture position / opening
        bl.SampleAperture.center = [
            0.,
            monochromator_distance + (sample_ap_distance - monochromator_distance) * np.cos(2 * theta0),
            (sample_ap_distance - monochromator_distance) * np.sin(2 * theta0),
        ]
        bl.SampleAperture.opening = [*bl.FrontEnd.opening[:2], -ap_size, ap_size]

        print(
            '#### Monochromator: E%d = %.01f keV; Θ = %.03f°; Rx = %.03f°\n'
            '#### Aperture: ΔE = ±%.01f eV; Δz = ±%.03f mm' % (
                ii,
                energy * 1e-3,
                np.degrees(theta0),
                np.degrees(pitch),
                d_energy,
                ap_size,
            )
        )

        # adjusting plots
        # rename all
        for plot in plts:
            plot.saveName = '%s-%.01fkeV.png' % (plot.title, energy * 1e-3)
            plot.caxis.offset = energy
            plot.caxis.limits = [energy - np.ceil(d_energy), energy + np.ceil(d_energy)]

        plts[3].yaxis.offset = np.tan(2 * theta0)
        plts[3].yaxis.limits = [
            ((sample_ap_distance - monochromator_distance) * np.sin(2 * theta0) - .8 * ap_size) / ((sample_ap_distance - monochromator_distance) * np.cos(2 * theta0)),
            ((sample_ap_distance - monochromator_distance) * np.sin(2 * theta0) + .8 * ap_size) / ((sample_ap_distance - monochromator_distance) * np.cos(2 * theta0))
        ]
        yield


# ###################################################### MAIN ##########################################################


if __name__ == '__main__':
    beamline = SKIF15()
    scan = energy_scan

    if show:
        beamline.glow(
            scale=[1e3, 1e5, 1e3],
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
