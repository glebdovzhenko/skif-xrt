from xrt.backends.raycing.oes_base import OE
from xrt.backends import raycing
import numpy as np


class BentLaueParaboloid(OE):
    """
    Reflective optical element in Laue geometry. 
    Takes two radii: Rx, Ry which can be positive or negative to create a circular surface.

    The surface equation is:
    z = 
    """

    # cl_plist = ("Rx", "Ry")
    # cl_local_z = """
    # float local_z(float8 cl_plist, int i, float x, float y)
    # {{
    #     return cl_plist.s0 * (1. - sqrt(1. - (x / cl_plist.s0) * (x / cl_plist.s0))) + cl_plist.s1 * (1. - sqrt(1. - (y / cl_plist.s1) * (y / cl_plist.s1)))

    # }}"""

    def __init__(self, *args, **kwargs):
        """
        Rx, Ry: bending radii in x and y directions. 
        Can be positive or negative, also np.inf or -np.inf.
        """
        # initializing fields
        self._Rx = np.inf
        self._Ry = np.inf
        self.R = np.inf
        self.r_for_refl = 'x'

        kwargs = self.__pop_kwargs(**kwargs)
        OE.__init__(self, *args, **kwargs)

    def __pop_kwargs(self, **kwargs):
        self.r_for_refl = kwargs.pop('r_for_refl', 'x')
        self.Rx = kwargs.pop('Rx', np.inf)
        self.Ry = kwargs.pop('Ry', np.inf)
        return kwargs
    
    @property
    def Rx(self):
        return self._Rx

    @Rx.setter
    def Rx(self, val):
        self._Rx = val
        if self.r_for_refl == 'x':
            self.R = val

    @property
    def Ry(self):
        return self._Ry

    @Ry.setter
    def Ry(self, val):
        self._Ry = val
        if self.r_for_refl == 'y':
            self.R = val

    def local_z(self, x, y):
        """Determines the surface of OE at (*x*, *y*) position."""
        if np.isinf(self.Rx) and np.isinf(self.Ry):
            return np.zeros_like(x)
        elif np.isinf(self.Rx):
            return self.Ry * (1. - np.sqrt(1. - (y / self.Ry) ** 2))
        elif np.isinf(self.Ry):
            return self.Rx * (1. - np.sqrt(1. - (x / self.Rx) ** 2))
        else:
            return self.Rx * (1. - np.sqrt(1. - (x / self.Rx) ** 2)) + self.Ry * (1. - np.sqrt(1. - (y / self.Ry) ** 2))

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        # [a, b, c] is the surface norm parallel to [-dz/dx, -dz/dy, 1.]
        if np.isinf(self.Rx) and np.isinf(self.Ry):
            c = 1.
        elif np.isinf(self.Rx):
            c = np.sqrt(1. - (y / self.Ry) ** 2)
        elif np.isinf(self.Ry):
            c = np.sqrt(1. - (x / self.Rx) ** 2)
        else:
            c = 1. / np.sqrt((self.Rx ** 2 * self.Ry ** 2 - x ** 2 * y ** 2) / 
                             ((self.Rx ** 2 - x ** 2) * (self.Ry ** 2 - y ** 2)))

        a = (-x / self.Rx) * c / np.sqrt(1. - (x / self.Rx) ** 2)
        b = (-y / self.Ry) * c / np.sqrt(1. - (y / self.Ry) ** 2)
        
        # Bragg norm is calculated as rotated surface norm
        if not self.alpha:
            bB, cB = c, -b
            return [a, bB, cB, a, b, c]

        if np.isinf(self.Rx) and np.isinf(self.Ry):
            if self.alpha:
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

    def pretty_R(self):
        if self.R < np.inf:
            return ('%.03f' % (self.R * 1e-3))
        else:
            return 'inf'


if __name__ == '__main__':
    from itertools import product
    import xrt.backends.raycing.materials as rm
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from xrt.backends.raycing.oes import BentLaueCylinder as BentLaueCylinder2

    cr = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', t=2.)

    m1 = BentLaueParaboloid(
        name='asfdsad',
        pitch=np.pi / 2.,
        roll=0.,
        yaw=0.,
        alpha=0.,
        material=(cr,),
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
    
    rs = [-10, 10, np.inf]
    for rx, ry in product(rs, repeat=2):
        ax = plt.figure('Rx=%f mm Ry = %f mm' % (rx, ry)).add_subplot(projection='3d')

        # Make the grid
        x, y, z = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10), [0.])

        m1.Rx = rx
        m1.Ry = ry
        m1.set_alpha(np.radians(-35.3))
        z = m1.local_z(x, y)

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
