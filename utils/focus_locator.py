import os
import pickle
import shutil
import matplotlib as mpl
import numpy as np

from xrt.backends.raycing import BeamLine
from xrt.backends.raycing import get_x, get_z
from xrt.backends.raycing.screens import Screen
from xrt.plotter import XYCAxis, XYCPlot

from utils.xrtutils import get_integral_breadth


class FocusLocator:
    def __init__(self, beam_name: str, data_dir: str,
                 x0: float = 0., z0: float = 0., ymin: float = 5e4, ymax: float = 1e5,
                 nscreens: int = 20, niterations: int = 4, cutoff=.1, axes=['z']):
        self.screen_fmt = '_FLS_%.03f'
        self.beam_fmt = '_FLSLocal_%03d'
        self.plot_fmt = self.screen_fmt

        self.x0 = x0
        self.z0 = z0

        self.beam_name = beam_name  # name of the beam to focus
        self.ymin = ymin
        self.ymax = ymax
        self.nscreens = nscreens
        self.niterations = niterations
        self.cutoff = cutoff
        self.axes = axes

        self.data_dir = data_dir
        self._subdir = os.path.join(self.data_dir, '_tmp')

        self.screen_fit = get_integral_breadth

    def _init_decorator(self, fn):
        def fn_(self_, *args, **kwargs):
            fn(self_, *args, **kwargs)
            self_._FLScreens = []
        return fn_

    def _reset_data_dir(self):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        if os.path.exists(self._subdir):
            shutil.rmtree(self._subdir)
        os.mkdir(self._subdir)

    def _reset_plots(self, bl, plts):
        for ii in range(len(plts) - 1, -1, -1):
            if self.plot_fmt.split('%')[0] in plts[ii].title:
                del plts[ii]

        plts.extend([
            XYCPlot(
                beam=self.beam_fmt % iscreen,
                title=self.plot_fmt % screen.center[1],
                persistentName=os.path.join(self._subdir, self.plot_fmt %
                                            screen.center[1] + '.pickle'),
                saveName=os.path.join(self._subdir, self.plot_fmt %
                                      screen.center[1] + '.png'),
                aspect='auto',
                xaxis=XYCAxis(label='$x$', unit='mm', data=get_x),
                yaxis=XYCAxis(label='$z$', unit='mm', data=get_z,
                              limits=[-.5, .5]))
            for iscreen, screen in enumerate(bl._FLScreens)
        ])

    @staticmethod
    def slice_parabola(a, b, c, m):
        m += 1.
        x0 = -b / (2. * c)
        a_ = a + m * (b * b / (4 * c) - a)
        d = np.sqrt(b * b - 4 * a_ * c)
        x1 = (-b - d) / (2. * c)
        x2 = (-b + d) / (2. * c)
        return x0, x1, x2

    def parabolic_fit(self, bl_, cutoff=.1):
        pos, x_size, z_size = [], [], []
        flux = None
        for f_name in (os.path.join(self._subdir, self.plot_fmt %
                                    screen.center[1] + '.pickle')
                       for screen in bl_._FLScreens):
            with open(f_name, 'rb') as f:
                f = pickle.load(f)
                x_size.append(self.screen_fit(f, 'x'))
                z_size.append(self.screen_fit(f, 'y'))
                pos.append(float(
                    os.path.basename(f_name).replace('.pickle', '').
                    replace('_FLS_', '')))

                if flux is None:
                    if hasattr(f, 'flux'):
                        flux = f.flux
                    elif hasattr(f, 'intensity'):
                        flux = f.intensity
        else:
            pos, x_size, z_size = np.array(pos), np.array(x_size), \
                np.array(z_size)
            ii = np.argsort(pos)
            pos, x_size, z_size = pos[ii], x_size[ii], z_size[ii]

        if 'z' in self.axes:
            pp_z = np.polynomial.polynomial.Polynomial.fit(pos, z_size, 2)
            coef_z = pp_z.convert().coef
            focus_z, ymin_z, ymax_z = self.slice_parabola(*coef_z, cutoff)
        else:
            ymin_z, ymax_z, focus_z = np.inf, -np.inf, None

        if 'x' in self.axes:
            pp_x = np.polynomial.polynomial.Polynomial.fit(pos, x_size, 2)
            coef_x = pp_x.convert().coef
            focus_x, ymin_x, ymax_x = self.slice_parabola(*coef_x, cutoff)
        else:
            ymin_x, ymax_x, focus_x = np.inf, -np.inf, None

        y_min, y_max = np.min([ymin_x, ymin_z]), np.max([ymax_x, ymax_z])

        fig, (ax1, ax2) = mpl.pyplot.subplots(2, 1)
        fig.suptitle(r'$\Phi$ = %f' % flux)
        ax1.plot(pos, z_size)
        if focus_z is not None:
            ax1.plot(pos, pp_z(pos))
            ax1.plot([focus_z, focus_z], [z_size.min(), z_size.max()], '--')
            ax1.text(focus_z, z_size.max(), 'Fz=%.01f mm' % focus_z)
        ax1.set_xlabel('Y position [mm]')
        ax1.set_ylabel('$\Delta Z$ [mm]')
        ax2.plot(pos, x_size)
        if focus_x is not None:
            ax2.plot(pos, pp_x(pos))
            ax2.plot([focus_x, focus_x], [x_size.min(), x_size.max()], '--')
            ax2.text(focus_x, x_size.max(), 'Fx=%.01f mm' % focus_x)
        ax2.set_xlabel('Y position [mm]')
        ax2.set_ylabel('$\Delta X$ [mm]')
        mpl.pyplot.tight_layout()
        fig.savefig(os.path.join(self.data_dir, 'fdist-%.03fm-%.03fm.png' %
                                 (np.min(pos * 1e-3), np.max(pos * 1e-3))))

        return ymin_z, ymax_z

    def beamline(self, klass):
        def flscreens_reset(self_, y_min, y_max, n):
            del self_._FLScreens[:]
            self_._FLScreens = [
                Screen(
                    bl=self_, name=self.screen_fmt % yy,
                    center=[self.x0, yy, self.z0]
                ) for yy in np.linspace(y_min, y_max, n)]

        assert issubclass(klass, BeamLine)

        klass.__init__ = self._init_decorator(klass.__init__)
        klass.flscreens_reset = flscreens_reset

        return klass

    def run_process(self, fn):
        def fn_(bl_):
            outDict_ = fn(bl_)
            for iscreen, screen in enumerate(bl_._FLScreens):
                outDict_[self.beam_fmt % iscreen] = \
                    screen.expose(beam=outDict_[self.beam_name])
            return outDict_
        return fn_

    def gnrtr(self, y_min=None, y_max=None, nscreens=None, niterations=None,
              cutoff=None):
        if y_min is not None:
            self.ymin = y_min
        if y_max is not None:
            self.ymax = y_max
        if nscreens is not None:
            self.nscreens = nscreens
        if niterations is not None:
            self.niterations = niterations
        if cutoff is not None:
            self.cutoff = cutoff

        def inner(fn):
            def fn_(bl_, plts_, *args, **kwargs):
                gn = fn(bl_, plts_, *args, **kwargs)
                for res in gn:
                    for ii in range(self.niterations):
                        bl_.flscreens_reset(
                            self.ymin,
                            self.ymax,
                            self.nscreens
                        )
                        self._reset_data_dir()
                        self._reset_plots(bl_, plts_)
                        yield res
                        self.ymin, self.ymax = self.parabolic_fit(
                            bl_,
                            self.cutoff
                        )
            return fn_

        return inner
