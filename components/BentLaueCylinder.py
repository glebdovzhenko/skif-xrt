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

        kwargs = self.__pop_kwargs(**kwargs)
        self.crossSectionInt = 0 if self.crossSection.startswith('circ') else 1
        OE.__init__(self, *args, **kwargs)
        if isinstance(self.R, (tuple, list)):
            self.R = self.get_Rmer_from_Coddington(self._R[0], self._R[1])

        # print(
        #     'R = %f, bendingOrientation = %s, convergingBend = %s' % (
        #     self.R, self.bendingOrientation, str(self.convergingBend))
        # )
        # print(self.cl_local_z)

    def __pop_kwargs(self, **kwargs):
        self.R = kwargs.pop('R', 1.0e4)
        self.crossSection = kwargs.pop('crossSection', 'parabolic')
        if not (self.crossSection.startswith('circ') or
                self.crossSection.startswith('parab')):
            raise ValueError('unknown crossSection!')
        self.bendingOrientation = kwargs.pop('bendingOrientation', 'meridional')
        return kwargs

    def upd_cl_local_z(self):
        self.cl_local_z = self.cl_local_z_fmt.format(
            '' if self.R > 0 else '-',
            '-' if self.R > 0 else '+',
            'y' if self.bendingOrientation == 'meridional' else 'x'
        )
        # print(self.R, '\n', self.bendingOrientation, '\n', self.cl_local_z)

    def local_z(self, x, y):
        """Determines the surface of OE at (*x*, *y*) position."""
        if self.bendingOrientation == 'meridional':
            var = y
        else:
            var = x

        if self.crossSection.startswith('circ'):  # 'circular'
            result = self.R - np.sqrt(self.R**2 - var**2) * np.sign(self.R)
        else:  # 'parabolic'
            result = var**2 / 2.0 / self.R

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
        
        # by this point we have [a, b, c] as norm vector with Abs == 1.
        # now we need to figure out the rotation.
        
        if not alpha:
            bB, cB = c, -b
            return [a, bB, cB, a, b, c]

        if np.isinf(R):
            if alpha:
                bB, cB = raycing.rotate_x(b, c, -self.sinalpha, -self.cosalpha)
            else:
                bB, cB = c, -b
            return [a, bB, cB, a, b, c]

        # this is the deformation (rotation) we need to apply to bragg norm
        cosrot = c
        sinrot = np.sqrt(a**2 + b**2)
        aRD = -b
        bRD = a
        cRD = np.zeros_like(x)
        
        # this is undeformed state of bragg norm
        aB = np.zeros_like(x)
        bB, cB = raycing.rotate_x(np.zeros_like(x), np.ones_like(x), 
                                  -self.sinalpha, -self.cosalpha)

        # rotation
        aB, bB, cB = (1. / (aRD**2 + bRD**2 + cRD**2)) * (aRD * (aB * aRD + bB * bRD + cB * cRD) + (-aRD * (bB * bRD + cB * cRD) + aB * (bRD**2 + cRD**2)) * cosrot + (bRD * cB - bB * cRD) * np.sqrt(aRD**2 + bRD**2 + cRD**2) * sinrot), \
            (1. / (aRD**2 + bRD**2 + cRD**2)) * (bRD * (aB * aRD + bB * bRD + cB * cRD) + (aRD**2 * bB - aB * aRD * bRD + cRD * (-bRD * cB + bB * cRD)) * cosrot - (aRD * cB - aB * cRD) * np.sqrt(aRD**2 + bRD**2 + cRD**2) * sinrot), \
             (1. / (aRD**2 + bRD**2 + cRD**2)) * (cRD * (aB * aRD + bB * bRD + cB * cRD) + (aRD**2 * cB - aB * aRD * cRD + bRD * (bRD * cB - bB * cRD)) * cosrot + (aRD * bB - aB * bRD) * np.sqrt(aRD**2 + bRD**2 + cRD**2) * sinrot)

        return [aB, bB, cB, a, b, c]


        # if alpha:
        #     bB, cB = raycing.rotate_x(b, c, -self.sinalpha, -self.cosalpha)
        # else:
        #     bB, cB = c, -b
        # return [a, bB, cB, a, b, c]

    # def _local_coord_system(self, x, y, R, alpha):
    #     """
    #     My own function for testing purposes.
    #     Eq for the surface: z = R - np.sign(R) * np.sqrt(R**2 - V**2) where
    #     V is y for meridional and
    #     V is x for sagittal bend.

    #     This gives us the normal vector [dz/dx, dz/dy, -1] as
    #     [0, np.sign(R) * y / np.sqrt(R**2 - y**2), -1] for meridional and 
    #     [np.sign(R) * x / np.sqrt(R**2 - x**2), 0, -1] for sagittal
    #     Which after normalization becomes
    #     [0, np.sign(R) * y / R, -np.sqrt(R**2 - y**2) / R] for meridional and
    #     [np.sign(R) * x / R, 0, -np.sqrt(R**2 - x**2) / R] for sagittal.

    #     """

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        return self.local_n_cylinder(x, y, self.R, self.alpha)

    def pretty_R(self):
        if self.R < np.inf:
            return ('%.03f' % (self.R * 1e-3))
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

    m2 = BentLaueCylinder2(
        name='asfdsad',
        pitch=np.pi / 2.,
        roll=0.,
        yaw=0.,
        alpha=0.,
        material=(cr,),
        R=np.inf,
        targetOpenCL='CPU',
        crossSection='circular'
    )

    # for orient in ('meridional', 'sagittal'):
    #     m1.bendingOrientation = orient
    
    #     for r in [3000, -3000, np.inf]:
    #         m1.R = r
    #         print(r, m1.R)
    
    #         xs, ys = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    #         zs = m1.local_z(xs, ys)
    
    #         fig = plt.figure('My class: R = %.01f m; %s' % (r, orient))
    #         ax = plt.axes(projection='3d')
    #         surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
    #                                linewidth=0, antialiased=False)
    #         plt.xlabel('local x')
    #         plt.ylabel('local y')
    
    # for r in [3000, -3000, np.inf]:
    #     m2.R = r
    #     print(r, m2.R)
    
    #     xs, ys = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    #     zs = m2.local_z(xs, ys)
    
    #     fig = plt.figure('Original class: R = %.01f m' % r)
    #     ax = plt.axes(projection='3d')
    #     surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
    #                            linewidth=0, antialiased=False)
    #     plt.xlabel('local x')
    #     plt.ylabel('local y')
    
    for rr in [-10., 10.]:
        for orient in ['meridional', 'sagittal']:
            for mi, name in zip((m1, m2), ('My', 'Original')):
                ax = plt.figure(name + ' ' + orient + ' R=%d mm' % rr).add_subplot(projection='3d')

                # Make the grid
                x, y, z = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10), [0.])

                mi.R = rr
                mi.set_alpha(np.radians(-35.3))
                mi.bendingOrientation = orient
                z = mi.local_z(x, y)

                # Make the direction data for the arrows
                u1, v1, w1, u2, v2, w2 = m1.local_n(x, y)

                ax.quiver(x, y, z, u1, v1, w1, normalize=False, color='r')
                ax.quiver(x, y, z, u2, v2, w2, normalize=False, color='g')

                plt.xlabel('local x')
                plt.ylabel('local y')
                ax.set_xlim(-5., 5.)
                ax.set_ylim(-5., 5.)
                ax.set_zlim(0., 10.)

    plt.show()
