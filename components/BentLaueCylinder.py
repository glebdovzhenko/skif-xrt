from xrt.backends.raycing.oes_base import OE
from xrt.backends import raycing
import numpy as np


class BentLaueCylinder(OE):
    """Simply bent reflective optical element in Laue geometry (duMond)."""

    cl_plist = ("crossSectionInt", "R")
    cl_local_z_fmt = """
    float local_z(float8 cl_plist, int i, float x, float y)
    {{
      if (cl_plist.s0 == 0)
        {{
          return {0} cl_plist.s1 {1} sqrt(cl_plist.s1*cl_plist.s1 - {2}*{2});
        }}
      else
        {{
          return {0} 0.5 * {2} * {2} / cl_plist.s1;
        }}
    }}"""

    def __init__(self, *args, **kwargs):
        """
        *R*: float or 2-tuple.
            Bending radius. Can be given as (*p*, *q*) for automatic
            calculation based the "Coddington" equations.
            Positive R for converging and negative for diverging.

        *crossSection*: str
            Determines the bending shape: either 'circular' or 'parabolic'.

        *bendingOrientation*: str
            Determines the axis around which the bending is done: x for 'meridional' and y for 'sagittal'

        """

        self._bendingOrientation = 'meridional'
        self._R = 1.0e4
        self._convergingBend = True

        kwargs = self.__pop_kwargs(**kwargs)
        self.crossSectionInt = 0 if self.crossSection.startswith('circ') else 1
        OE.__init__(self, *args, **kwargs)
        if isinstance(self.R, (tuple, list)):
            self.R = self.get_Rmer_from_Coddington(self._R[0], self._R[1])
        else:
            self.R = self._R
        # print(
        #     'R = %f, bendingOrientation = %s, convergingBend = %s' % (
        #     self.R, self.bendingOrientation, str(self.convergingBend))
        # )
        # print(self.cl_local_z)

    def __pop_kwargs(self, **kwargs):
        self._R = kwargs.pop('R', 1.0e4)
        self.crossSection = kwargs.pop('crossSection', 'parabolic')
        if not (self.crossSection.startswith('circ') or
                self.crossSection.startswith('parab')):
            raise ValueError('unknown crossSection!')
        self.bendingOrientation = kwargs.pop('bendingOrientation', 'meridional')
        return kwargs

    def upd_cl_local_z(self):
        self.cl_local_z = self.cl_local_z_fmt.format(
            '' if self.convergingBend else '-',
            '-' if self.convergingBend else '+',
            'y' if self.bendingOrientation == 'meridional' else 'x'
        )

    def local_z(self, x, y):
        """Determines the surface of OE at (*x*, *y*) position."""
        if self.bendingOrientation == 'meridional':
            var = y
        else:
            var = x

        if self.crossSection.startswith('circ'):  # 'circular'
            result = self.R - np.sqrt(self.R**2 - var**2)
        else:  # 'parabolic'
            result = var**2 / 2.0 / self.R

        if not self.convergingBend:
            result *= -1

        return result

    @property
    def bendingOrientation(self):
        return self._bendingOrientation

    @bendingOrientation.setter
    def bendingOrientation(self, val):
        if val not in ('meridional', 'sagittal'):
            raise ValueError('unknown bendingOrientation!')
        else:
            self._bendingOrientation = val
            self.upd_cl_local_z()

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, val):
        self._R = val
        if self._R < 0:
            self._convergingBend = False
            self._R *= -1.
        elif self._R > 0:
            self._convergingBend = True
        self.upd_cl_local_z()

    @property
    def convergingBend(self):
        return self._convergingBend

    @convergingBend.setter
    def convergingBend(self, val):
        self._convergingBend = val
        self.upd_cl_local_z()

    def local_n_cylinder(self, x, y, R, alpha):
        """The main part of :meth:`local_n`. It introduces two new arguments
        to simplify the implementation of :meth:`local_n` in the derived class
        :class:`GroundBentLaueCylinder`."""
        if self.bendingOrientation == 'meridional':
            a = np.zeros_like(x)  # -dz/dx
            b = -y / R  # -dz/dy
        else:
            a = -x / R
            b = np.zeros_like(y)

        if not self.convergingBend:
            a *= -1.
            b *= -1.

        if self.crossSection.startswith('circ'):  # 'circular'
            if self.bendingOrientation == 'meridional':
                c = (R ** 2 - y ** 2) ** 0.5 / R
            else:
                c = (R ** 2 - x ** 2) ** 0.5 / R

        elif self.crossSection.startswith('parab'):  # 'parabolic'
            norm = (b**2 + 1)**0.5
            b /= norm
            c = 1. / norm
        else:
            raise ValueError('unknown crossSection!')
        if alpha:
            bB, cB = raycing.rotate_x(b, c, -self.sinalpha, -self.cosalpha)
        else:
            bB, cB = c, -b
        return [a, bB, cB, a, b, c]

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        return self.local_n_cylinder(x, y, self.R, self.alpha)

    def pretty_R(self):
        if self.R < np.inf:
            return ('' if self.convergingBend else '-') + ('%.01f' % (self.R * 1e-3))
        else:
            return 'inf'


if __name__ == '__main__':
    import xrt.backends.raycing.materials as rm
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from xrt.backends.raycing.oes import BentLaueCylinder as BentLaueCylinder2

    cr = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', t=2.)

    m1 = BentLaueCylinder(
        name='asfdsad',
        pitch=np.pi / 2.,
        roll=0.,
        yaw=0.,
        alpha=0.,
        material=(cr,),
        R=np.inf,
        targetOpenCL='CPU',
    )

    for orient in ('meridional', 'sagittal'):
        m1.bendingOrientation = orient

        for r in [3000, -3000]:
            m1.R = r
            print(r, m1.R, m1.convergingBend)

            xs, ys = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
            zs = m1.local_z(xs, ys)

            fig = plt.figure('R = %.01f m; %s' % (r, orient))
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            plt.xlabel('local x')
            plt.ylabel('local y')
    plt.show()
