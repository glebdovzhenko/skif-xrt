from typing import List
from xrt.backends.raycing.sources import GeometricSource, Beam
from xrt.backends.raycing import BeamLine
from xrt.backends.raycing.materials import Material
from components import PrismaticLens
from xrt.backends.raycing.screens import Screen
from xrt.plotter import XYCAxis, XYCPlot
from xrt.backends.raycing import get_x, get_z
from utils.xrtutils import get_integral_breadth
from utils.focus_locator import FocusLocator


import xrt.backends.raycing.run as rrun
import xrt.runner as xrtrun

import numpy as np
import os
import shutil
import pickle

import matplotlib as mpl
mpl.use('agg')


data_dir = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'tmp')

# crl_mat = Material('C', rho=2.15, kind='lens')
# crl_mat = Material('Al', rho=2.7, kind='lens')
crl_mat = Material('Be', rho=1.848, kind='lens')
crl_y_t = 1.228  # mm
crl_y_g = 1.228  # mm
crl_L = 270.     # mm
en = 30000.      # eV

mul = 1.5
crl_y_g *= mul
crl_y_t *= mul
crl_L *= mul * mul

focal_dist = 14000.  # mm
# crl_y_g = focal_dist * crl_L * \
#     np.real(1. - crl_mat.get_refractive_index(en)) / crl_y_t
# focal_dist_calc = crl_y_g * crl_y_t / \
#     (crl_L * np.real(1. - crl_mat.get_refractive_index(en)))
# focal_dist = focal_dist_calc

FL = FocusLocator(
    beam_name='BeamFocused',
    data_dir=os.path.join(os.getenv('BASE_DIR'), 'datasets', 'tmp'),
    axes=['z']
)


@FL.beamline
class CrocTestBL(BeamLine):
    def __init__(self, azimuth=0, height=0, alignE='auto'):
        super().__init__(azimuth, height, alignE)

        self.name = 'Croc Lens Test BL'

        self.GS = GeometricSource(
            name=r"Gaussian Source",
            bl=self,
            center=[0, 0, 0],
            distE='lines',
            energies=[en],
            distx='normal',
            disty='flat',
            distz='normal',
            distxprime='flat',
            distzprime='flat',
            dx=.455,
            dy=0.,
            dz=.027,
            dxprime=1e-3,
            dzprime=1e-4,
        )

        self.LensMat = crl_mat

        print('Lens params: y_t: %.03f, y_g: %.03f, L: %.03f' %
              (crl_y_t, crl_y_g, crl_L))

        self.LensStack = PrismaticLens.make_stack(
            # int(crl_L)
            L=crl_L, N=int(crl_L), d=crl_y_t, g_last=0.0, g_first=crl_y_g,
            bl=self,
            center=[0., 2. * focal_dist, 0],
            material=self.LensMat,
            limPhysX=[-100., 100.],
            limPhysY=[-20., 20.],
        )

        self.SourceScreen = Screen(
            bl=self,
            name=r"Source",
            center=[0, 1., 0],
        )
        self.PreLensScreen = Screen(
            bl=self,
            name=r"Before lens",
            center=[0, 2. * focal_dist - 1., 0],
        )
        self.PostLensScreen = Screen(
            bl=self,
            name=r"After lens",
            center=[0, 2. * focal_dist + crl_L + 1., 0],
        )
        self.FocusingScreen = Screen(
            bl=self,
            name=r"Focus",
            center=[0, 4. * focal_dist, 0],
        )


@FL.run_process
def run_process(bl: CrocTestBL):
    outDict = dict()

    outDict['BeamSourceGlobal'] = bl.GS.shine()
    outDict['BeamM1Local'] = bl.SourceScreen.expose(
        beam=outDict['BeamSourceGlobal'])
    outDict['BeamM2Local'] = bl.PreLensScreen.expose(
        beam=outDict['BeamSourceGlobal'])

    beamIn = outDict['BeamSourceGlobal']
    for ilens, lens in enumerate(bl.LensStack):
        lglobal, llocal1, llocal2 = lens.double_refract(beamIn, needLocal=True)
        strl = '_{0:02d}'.format(ilens)
        outDict['BeamLensGlobal'+strl] = lglobal
        outDict['BeamLensLocal1'+strl] = llocal1
        outDict['BeamLensLocal2'+strl] = llocal2

        llocal2a = Beam(copyFrom=llocal2)
        llocal2a.absorb_intensity(beamIn)
        outDict['BeamLensLocal2a'+strl] = llocal2a
        beamIn = lglobal

    outDict['BeamM3Local'] = bl.PostLensScreen.expose(beam=beamIn)
    outDict['BeamM4Local'] = bl.FocusingScreen.expose(beam=beamIn)
    outDict['BeamFocused'] = beamIn

    bl.prepare_flow()
    return outDict


rrun.run_process = run_process


plots = [
    XYCPlot(beam='BeamM1Local', title='source', aspect='auto',
            xaxis=XYCAxis(label='$x$', unit='mm', data=get_x),
            yaxis=XYCAxis(label='$z$', unit='mm', data=get_z),
            persistentName=os.path.join(data_dir, 'BeamAtSource.pickle')),
    XYCPlot(beam='BeamM2Local', title='before_lens', aspect='auto',
            xaxis=XYCAxis(label='$x$', unit='mm', data=get_x),
            yaxis=XYCAxis(label='$z$', unit='mm', data=get_z)),
    XYCPlot(beam='BeamM3Local', title='after_lens', aspect='auto',
            xaxis=XYCAxis(label='$x$', unit='mm', data=get_x),
            yaxis=XYCAxis(label='$z$', unit='mm', data=get_z)),
    XYCPlot(beam='BeamM3Local', title='focal_point', aspect='auto',
            xaxis=XYCAxis(label='$x$', unit='mm', data=get_x),
            yaxis=XYCAxis(label='$z$', unit='mm', data=get_z))
]


def onept(bl: CrocTestBL, plots: List):
    def slice_parabola(a, b, c, m):
        m += 1.
        x0 = -b / (2. * c)
        a_ = a + m * (b * b / (4 * c) - a)
        d = np.sqrt(b * b - 4 * a_ * c)
        x1 = (-b - d) / (2. * c)
        x2 = (-b + d) / (2. * c)
        return x0, x1, x2

    # updating beamline params
    ymin, ymax = 2.1 * focal_dist, 6. * focal_dist
    for _ in range(4):
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.mkdir(data_dir)

        bl.flscreens_reset(y_min=ymin, y_max=ymax, n=20)
        for ii in range(len(plots) - 1, -1, -1):
            if '_FLS_' in plots[ii].title:
                del plots[ii]

        plots.extend([
            XYCPlot(
                beam=FL.beam_fmt % iscreen,
                title=FL.plot_fmt % screen.center[1],
                persistentName=os.path.join(
                    data_dir, FL.plot_fmt % screen.center[1] + '.pickle'),
                saveName=os.path.join(data_dir, FL.plot_fmt %
                                      screen.center[1] + '.png'),
                aspect='auto',
                xaxis=XYCAxis(label='$x$', unit='mm', data=get_x),
                yaxis=XYCAxis(label='$z$', unit='mm', data=get_z, limits=[-.5, .5]))
            for iscreen, screen in enumerate(bl._FLScreens)
        ])

        yield

        # calculating focus position and size
        pos, y_size = [], []
        f_names = (os.path.join(data_dir, FL.plot_fmt %
                   screen.center[1] + '.pickle') for screen in bl._FLScreens)
        for f_name in f_names:
            with open(f_name, 'rb') as f:
                y_size.append(get_integral_breadth(pickle.load(f), 'y'))
                pos.append(float(os.path.basename(f_name).replace(
                    '.pickle', '').replace('_FLS_', '')))
        else:
            pos, y_size = np.array(pos), np.array(y_size)
            ii = np.argsort(pos)
            pos, y_size = pos[ii], y_size[ii]

        pp = np.polynomial.polynomial.Polynomial.fit(pos, y_size, 2)
        coef = pp.convert().coef
        focus, ymin, ymax = slice_parabola(*coef, 0.1)

        fig = mpl.pyplot.figure()
        ax = fig.add_subplot()
        ax.plot(pos, y_size)
        ax.plot(pos, pp(pos))
        ax.plot([focus, focus], [y_size.min(), y_size.max()], '--')
        ax.text(focus, y_size.max(), 'F=%.01f mm' % focus)
        ax.set_xlabel('Y position [mm]')
        ax.set_ylabel('Beam integral breadth [mm]')
        fig.savefig(os.path.join(data_dir, '..', 'fdist%d.png' % _))
    else:
        # calculating gain
        focus_size = coef[0] - coef[1]**2 / (4. * coef[2])
        with open(f_name, 'rb') as f:
            f = pickle.load(f)
            focus_flux = f.intensity
        with open(os.path.join(data_dir, 'BeamAtSource.pickle'), 'rb') as f:
            f = pickle.load(f)
            source_flux = f.intensity
            source_size = get_integral_breadth(f, 'y')

        source_projection = source_size + 2. * focus * 1e-4
        gain = (focus_flux * source_projection) / (source_flux * focus_size)
        print('Focus  | size: %.03f | flux %.01f | distance %.01f' %
              (focus_size, focus_flux, focus))
        print('Source | size: %.03f | flux %.01f' % (source_size, source_flux))
        print('Projected source size: %.03f' % source_projection)
        print('Gain %.01f' % gain)
        ax.text(focus, y_size.max(), 'F=%.01f mm Gain = %.01f' % (focus, gain))
        fig.savefig(os.path.join(data_dir, '..', 'fdist%d.png' % _))


@FL.gnrtr(50e3, 70e3, 20)
def onept2(bl: CrocTestBL, plots: List):
    yield


if __name__ == '__main__':
    beamline = CrocTestBL()
    scan = onept2
    show = False
    repeats = 1

    if show:
        beamline.glow(
            scale=[1e1, 1e4, 1e4],
            centerAt=r'Lens_03_Exit',
            generator=scan,
            generatorArgs=[beamline, plots],
            startFrom=1
        )
    else:
        xrtrun.run_ray_tracing(
            beamLine=beamline,
            plots=plots,
            repeats=repeats,
            backend=r"raycing",
            generator=scan,
            generatorArgs=[beamline, plots]
        )
