from ast import Num
from xrt.backends.raycing.oes_base import OE
from xrt.backends import raycing
import numpy as np
from components.BentLaueParaboloid import BentLaueParaboloid
from components.bump_eqs import crs, crn, crbn
from components.bump_eqs import crs_x, crn_x, crbn_x, crs_y, crn_y, crbn_y, crs_xy, crn_xy, crbn_xy



class BentLaueParaboloidWithBump(BentLaueParaboloid):
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
        BentLaueParaboloid.__init__(self, *args, **kwargs)

        self.bump_pars = {
            'Cx': 0.,
            'Cy': 0.,
            'Sx': 1.,
            'Sy': 1., 
            'Axy': 3., 
        }

    def local_z(self, x, y):
        """Determines the surface of OE at (*x*, *y*) position."""
        # xS, yS, Cx, Cy, Rx, Ry, Sx, Sy, Axy, Chi
        if np.isinf(self.Rx) and np.isinf(self.Ry):
            return crs_xy(x, y, Rx=self._Rx, Ry=self._Ry, **self.bump_pars)
        elif np.isinf(self.Rx):
            return crs_y(x, y, Rx=self._Rx, Ry=self._Ry, **self.bump_pars)
        elif np.isinf(self.Ry):
            return crs_x(x, y, Rx=self._Rx, Ry=self._Ry, **self.bump_pars)
        else:
            return crs(x, y, Rx=self._Rx, Ry=self._Ry, **self.bump_pars)

    def local_n(self, x, y):
        """Determines the normal vector of OE at (x, y) position."""
        # [a, b, c] is the surface norm parallel to [-dz/dx, -dz/dy, 1.]
        if np.isinf(self.Rx) and np.isinf(self.Ry):
            a, b, c = crn_xy(x, y, Rx=self._Rx, Ry=self._Ry, **self.bump_pars)
        elif np.isinf(self.Rx):
            a, b, c = crn_y(x, y, Rx=self._Rx, Ry=self._Ry, **self.bump_pars)
        elif np.isinf(self.Ry):
            a, b, c = crn_x(x, y, Rx=self._Rx, Ry=self._Ry, **self.bump_pars)
        else:
            a, b, c = crn(x, y, Rx=self._Rx, Ry=self._Ry, **self.bump_pars)
        
        # Bragg norm is calculated as rotated surface norm
        if not self.alpha:
            bB, cB = c, -b
            return [a, bB, cB, a, b, c]
        
        sa, ca = np.sin(self.alpha + np.pi / 2), np.cos(self.alpha + np.pi / 2)
        if np.isinf(self.Rx) and np.isinf(self.Ry):
            aB, bB, cB = crbn_xy(x, y, Rx=self._Rx, Ry=self._Ry, SinAlpha=sa, CosAlpha=ca, **self.bump_pars)
        elif np.isinf(self.Rx):
            aB, bB, cB = crbn_y(x, y, Rx=self._Rx, Ry=self._Ry, SinAlpha=sa, CosAlpha=ca, **self.bump_pars)
        elif np.isinf(self.Ry):
            aB, bB, cB = crbn_x(x, y, Rx=self._Rx, Ry=self._Ry, SinAlpha=sa, CosAlpha=ca, **self.bump_pars)
        else:
            aB, bB, cB = crbn(x, y, Rx=self._Rx, Ry=self._Ry, SinAlpha=sa, CosAlpha=ca, **self.bump_pars)
        
        nan_ii = np.isnan(aB) | np.isnan(bB) | np.isnan(cB)
        aB[nan_ii] = a[nan_ii]
        bB[nan_ii], cB[nan_ii] = raycing.rotate_x(b[nan_ii], c[nan_ii], -self.sinalpha, -self.cosalpha)

        return [aB, bB, cB, a, b, c]


if __name__ == '__main__':
    from itertools import product
    import os
    import xrt.backends.raycing.materials as rm
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from xrt.backends.raycing.oes import BentLaueCylinder as BentLaueCylinder2

    wd = r'/Users/glebdovzhenko/Yandex.Disk.localized/Dev/skif-xrt/datasets/bump'
 
    cr = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', t=2.)

    m1 = BentLaueParaboloidWithBump(
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

    # for r in [-10, 10, np.inf]:
    #     ax = plt.figure('Original class R = %f mm' % (r, )).add_subplot(projection='3d')

    #     # Make the grid
    #     x, y, z = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10), [0.])
        
    #     m2.R = r
    #     m2.set_alpha(np.radians(-35.3))
    #     z = m2.local_z(x, y)

    #     # Make the direction data for the arrows
    #     u1, v1, w1, u2, v2, w2 = m2.local_n(x, y)
        
    #     ax.quiver(x, y, z, u1, v1, w1, normalize=False, color='r')
    #     ax.quiver(x, y, z, u2, v2, w2, normalize=False, color='g')

    #     plt.xlabel('local x')
    #     plt.ylabel('local y')
    #     ax.set_xlim(-5., 5.)
    #     ax.set_ylim(-5., 5.)
    #     ax.set_zlim(0., 10.)

    rs = [-10, 10, np.inf]
    for rx, ry in product(rs, repeat=2):
        ax = plt.figure('Rx=%f mm Ry = %f mm' % (rx, ry)).add_subplot(projection='3d')

        # Make the grid
        x, y, z = np.meshgrid(np.linspace(-5, 5, 11), np.linspace(-5, 5, 11), [0.])

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
    
    # # setting bump params from files 
    # for fpath in filter(lambda x: x[-4:] == '.npy', os.listdir(wd)):
    #     params = np.load(os.path.join(wd, fpath))
    #     m1.bump_pars = {
    #         'Cx': params[0],
    #         'Cy': params[1],
    #         'Sx': params[2],
    #         'Sy': params[3], 
    #         'Axy': params[4], 
    #     }
    #     ax = plt.figure(fpath).add_subplot(projection='3d')

    #     # Make the grid
    #     x, y, z = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10), [0.])

    #     m1.Rx = np.inf
    #     m1.Ry = np.inf
    #     m1.set_alpha(np.radians(-35.3))
    #     z = m1.local_z(x, y)

    #     # Make the direction data for the arrows
    #     u1, v1, w1, u2, v2, w2 = m1.local_n(x, y)
        
    #     ax.quiver(x, y, z, u1, v1, w1, normalize=False, color='r')
    #     ax.quiver(x, y, z, u2, v2, w2, normalize=False, color='g')

    #     plt.xlabel('local x')
    #     plt.ylabel('local y')
    #     ax.set_xlim(-5., 5.)
    #     ax.set_ylim(-5., 5.)
    #     ax.set_zlim(0., 10.)

    plt.show()
