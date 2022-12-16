import numpy as np

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
from components import CrystalSiPrecalc
from params.sources import ring_kwargs, wiggler_1_5_kwargs
from params.params_NSTU_SCW import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle, \
    monochromator_distance, monochromator_z_offset, monochromator_x_lim, monochromator_y_lim, \
    crl_distance


# ################################################## SIM PARAMETERS ####################################################


show = False
repeats = 10

bending_orientation = 'sagittal' # 'meridional', 'sagittal'
useTT = True


"scan1"
scan = 'focus_spot_scan'
subdir = 'img/scan5'
energies = [30. * 1e3]
de_over_e = 0.0012
plot_z_lim = 1e-2
plot_zpr_lim = 1e-2
mono = False


# ################################################# SETUP PARAMETERS ###################################################


""" Monochromator """
monochromator_alignment_energy = energies[0]
monochromator_c1_alpha = np.radians(-30.0)
monochromator_c1_thickness = .2
monochromator_c2_alpha = np.radians(30.0)
monochromator_c2_thickness = .2


# #################################################### MATERIALS #######################################################


cr_si_1 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=useTT, t=monochromator_c1_thickness, 
                       database='/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/components/Si111ref_sag.csv')
cr_si_2 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=useTT, t=monochromator_c2_thickness, 
                       database='/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/components/Si111ref_sag.csv')


# #################################################### BEAMLINE ########################################################


class BENDY(raycing.BeamLine):
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
            bendingOrientation=bending_orientation,
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
            name=r"Crystal 1 Monitor",
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
            bendingOrientation=bending_orientation,
            R=np.inf,
            targetOpenCL='CPU',
            limPhysY=monochromator_y_lim,
            limOptY=monochromator_y_lim,
            limPhysX=monochromator_x_lim,
            limOptX=monochromator_x_lim,
        )
        self.MonochromatorCr2.ucl = mcl.XRT_CL(r'materials.cl', targetOpenCL='CPU')

        self.Cr2Monitor = rscreens.Screen(
            bl=self,
            name=r"Crystal 2 Monitor",
            center=[0, crl_distance - 10, monochromator_z_offset],
        )


# ################################################# BEAM TOPOLOGY ######################################################


def run_process(bl: BENDY):

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

    beam_mon2 = bl.Cr2Monitor.expose(
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
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title="FE-XprZpr",
        xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title="FE-ZZpr",
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title="FE-XXpr",
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        aspect='auto'),

    # Monitor between crystals
    xrtplot.XYCPlot(
        beam='BeamMonitor1Local',
        title='C1-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor1Local',
        title='C1-XprZpr',
        xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor1Local',
        title='C1-ZZpr',
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor1Local',
        title='C1-XXpr',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        aspect='auto'),

    # Exit monitor
    xrtplot.XYCPlot(
        beam='BeamMonitor2Local',
        title='C2-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor2Local',
        title='C2-XprZpr',
        xaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor2Local',
        title='C2-ZZpr',
        xaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        yaxis=xrtplot.XYCAxis(label=r'$z^{\prime}$', unit='', data=raycing.get_zprime),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor2Local',
        title='C2-XXpr',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$x^{\prime}$', unit='', data=raycing.get_xprime),
        aspect='auto'),
]


# ############################################### SEQUENCE GENERATOR ###################################################


def print_positions(bl: BENDY):
    print('#' * 20, bl.name, '#' * 20)

    for element in (bl.SuperCWiggler, bl.FrontEnd,
                    bl.MonochromatorCr1, bl.Cr1Monitor, bl.MonochromatorCr2,
                    bl.Cr2Monitor):
        print('#' * 5, element.name, 'at', element.center)

    print('#' * (42 + len(bl.name)))


def align_energy(bl: BENDY, en, d_en):
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

    # bl.MonochromatorCr1.pitch += np.pi
    # bl.MonochromatorCr2.pitch += np.pi

    print_positions(bl)


def upd_plots(plts, bl: BENDY, en):
    global de_over_e, plot_z_lim, plot_zpr_lim
    """
    'FE-XZ',"FE-XprZpr","FE-ZZpr","FE-XXpr",
    'C1-XZ','C1-XprZpr','C1-ZZpr','C1-XXpr',
    'C2-XZ','C2-XprZpr','C2-ZZpr','C2-XXpr'
    """
    for plot in plts:
        # adjusting energy limits and offset
        plot.caxis.offset = en
        plot.caxis.limits = [en * (1. - de_over_e), en * (1. + de_over_e)]

        # By default no plots need offset
        plot.xaxis.offset = 0.
        plot.yaxis.offset = 0.

        # Except for all z^{\prime} axes that are between C1 & C2.
        # They have to be offset by 1st crystal scattering direction.
        if plot.title in ('C1-XprZpr', 'C1-ZZpr'):
            plot.yaxis.offset = bl.Cr1Monitor.center[2] / (bl.Cr1Monitor.center[1] - bl.MonochromatorCr1.center[1])

        # Setting z^{\prime} limits
        if 'Zpr' in plot.title:
            plot.yaxis.limits = [plot.yaxis.offset - plot_zpr_lim, plot.yaxis.offset + plot_zpr_lim]

        # setting z limits
        if 'XZ' in plot.title:
            plot.yaxis.limits = [plot.yaxis.offset - plot_z_lim, plot.yaxis.offset + plot_z_lim]

        if 'ZZpr' in plot.title:
            plot.xaxis.limits = [plot.xaxis.offset - plot_z_lim, plot.xaxis.offset + plot_z_lim]

        # setting X limits for sagittal focusing
        if plot.title in ('C2-XXpr', 'C2-XZ'):
            plot.xaxis.limits = [-1., 1.]


def set_de_over_e(radius):
    global de_over_e, plot_z_lim, plot_zpr_lim

    if radius < 4.e3:
        de_over_e = 1.5e4 / 70.e3
    elif 4.e3 <= radius < 8.e3:
        de_over_e = 4.5e3 / 70.e3
    elif 8.e3 <= radius < 15.e3:
        de_over_e = 1.5e3 / 70.e3
    elif 15.e3 <= radius < 21.e3:
        de_over_e = 7.5e2 / 70.e3
    elif 21.e3 <= radius < 28.e3:
        de_over_e = 3.e2 / 70.e3
    elif 28.e3 <= radius < 30.e3:
        de_over_e = 2.e2 / 70.e3
    elif 30.e3 <= radius < 39.e3:
        de_over_e = 7.5e1 / 70.e3
    elif 39.e3 <= radius < 52.e3:
        de_over_e = 1.5e2 / 70.e3
    elif 52.e3 <= radius < 100.e3:
        de_over_e = 3.e2 / 70.e3
    elif 100.e3 <= radius:
        de_over_e = 6.e2 / 70.e3

    if 3.e4 <= radius:
        plot_zpr_lim = 1e-3
    elif 1.5e4 <= radius < 3.e4:
        plot_zpr_lim = 2e-3
    elif radius < 1.5e4:
        plot_zpr_lim = 7e-3

    plot_z_lim = 5.


def scan1(plts, bl: BENDY):

    energy = energies[0]
    set_de_over_e(np.abs(bl.MonochromatorCr1.R))
    align_energy(bl, energy, de_over_e)

    for r in np.arange(2., 6., .1) * 1e3:
        bl.MonochromatorCr1.R = r
        bl.MonochromatorCr2.R = r

        if bending_orientation == 'meridional':
            set_de_over_e(np.abs(bl.MonochromatorCr1.R))
        else:
            set_de_over_e(70000.)

        align_energy(bl, energy, de_over_e)

        for plot in plts:
            plot.saveName = '%s/%s-%dkeV-%.01fmm-%sm-%sm.png' % (
                subdir, plot.title,
                int(energy * 1e-3),
                monochromator_c1_thickness,
                bl.MonochromatorCr1.pretty_R(),
                bl.MonochromatorCr2.pretty_R()
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        upd_plots(plts, bl, energy)
        yield


def focus_spot_scan(plts, bl: BENDY):
    """
    [[2.0, 8.011777], [2.1, 8.671524], [2.2, 9.316851], [2.3, 10.012308], [2.4, 10.758524], [2.5, 11.450532], 
    [2.6, 12.171743], [2.7, 12.950061], [2.8, 13.917753], [2.9, 15.543408], [3.0, 15.286669], [3.1, 16.499600], 
    [3.2, 17.453257], [3.3, 18.170581], [3.4, 19.335992], [3.5, 20.281856], [3.6, 21.352264], [3.7, 22.472150], 
    [3.8, 23.585522], [3.9, 24.803094], [4.0, 26.052875], [4.1, 27.263725], [4.2, 28.758247], [4.3, 30.016872], 
    [4.4, 31.612202], [4.5, 33.090912], [4.6, 34.705426], [4.7, 36.392103], [4.8, 38.251350], [4.9, 39.995497], 
    [5.0, 41.911412], [5.1, 44.243416], [5.2, 46.200090], [5.3, 48.454464], [5.4, 51.218212], [5.5, 53.538067], 
    [5.6, 56.236627], [5.7, 59.592210], [5.8, 62.261050], [5.9, 65.341203], [6.0, 69.480730]]
    """

    f_pos = np.array([[2.0, 9.783779], [2.1, 10.428126], [2.2, 11.091144], [2.3, 11.777164], [2.4, 12.480913], 
                      [2.5, 13.209525], [2.6, 13.960662], [2.7, 14.734530], [2.8, 15.536195], [2.9, 16.366854], 
                      [3.0, 17.227896], [3.1, 18.116948], [3.2, 19.037449], [3.3, 19.980908], [3.4, 20.968621], 
                      [3.5, 21.989276], [3.6, 23.054367], [3.7, 24.161776], [3.8, 25.310650], [3.9, 26.514691], 
                      [4.0, 27.751589], [4.1, 29.047774], [4.2, 30.406044], [4.3, 31.820153], [4.4, 33.304696], 
                      [4.5, 34.844255], [4.6, 36.482382], [4.7, 38.186472], [4.8, 39.965321], [4.9, 41.838138], 
                      [5.0, 43.806718], [5.1, 45.944693], [5.2, 48.128957], [5.3, 50.444744], [5.4, 52.984118], 
                      [5.5, 55.556352], [5.6, 58.345372], [5.7, 61.420888], [5.8, 64.571991], [5.9, 67.906893], 
                      [6.0, 71.784676]])

    energy = energies[0]
    set_de_over_e(np.abs(bl.MonochromatorCr1.R))
    align_energy(bl, energy, de_over_e)

    for r, offset in f_pos * 1e3:
        bl.MonochromatorCr1.R = r
        bl.MonochromatorCr2.R = r

        if bending_orientation == 'meridional':
            set_de_over_e(np.abs(bl.MonochromatorCr1.R))
        else:
            set_de_over_e(70000.)

        align_energy(bl, energy, de_over_e)
        
        bl.Cr2Monitor.center = [
                bl.MonochromatorCr2.center[0],
                bl.MonochromatorCr2.center[1] + offset,
                bl.MonochromatorCr2.center[2]
        ]

        for plot in plts:
            plot.saveName = '%s/%s-%dkeV-%.01fmm-%sm-%sm.png' % (
                subdir, plot.title,
                int(energy * 1e-3),
                monochromator_c1_thickness,
                bl.MonochromatorCr1.pretty_R(),
                bl.MonochromatorCr2.pretty_R()
            )
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

        upd_plots(plts, bl, energy)
        yield



    yield

# ###################################################### MAIN ##########################################################


if __name__ == '__main__':
    beamline = BENDY()
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
