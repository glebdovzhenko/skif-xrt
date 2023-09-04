import numpy as np

import xrt.backends.raycing.oes as roe


class PrismaticLens(roe.Plate):
    r"""
    """

    hiddenMethods = roe.Plate.hiddenMethods + ['double_refract']
    cl_plist = ("half_opening", "z1max", "z2max", "k1", "k2")
    cl_local_z = """
    float local_z1(float8 cl_plist, float x, float y)
    {
        float res;
        res = (abs(y) - cl_plist.s0) * cl_plist.s3;
        if (abs(y) < cl_plist.s0) res = 0.;
        if (res > cl_plist.s1) res = cl_plist.s1;
        return res;
    }

    float local_z2(float8 cl_plist, float x, float y)
    {
        float res;
        res = (abs(y) - cl_plist.s0) * cl_plist.s4);
        if (abs(y) < cl_plist.s0) res = 0.;
        if (res > cl_plist.s2) res = cl_plist.s2;
        return res;
    }

    float local_z(float8 cl_plist, int i, float x, float y)
    {
        float res=0;
        if (i == 1)
            res = local_z1(cl_plist, x, y);
        if (i == 2)
            res = local_z2(cl_plist, x, y);
        return res;
    }
    """

    cl_local_n = """
    float3 local_n1(float8 cl_plist, float x, float y)
    {
        float3 res;
        res.s0 = 0.;
        res.s1 = 0.;
        res.s2 = 1.;
        
        if (abs(y) > cl_plist.s0) res.s1 = -sign(y) * cl_plist.s3;

        float z = (abs(y) - cl_plist.s0) * cl_plist.s3;
        if (z > cl_plist.s1)
        {
            res.s0 = 0;
            res.s1 = 0;
        }
        res.s2 = 1.;
        return normalize(res);
    }

    float3 local_n2(float8 cl_plist, float x, float y)
    {
        float3 res;
        res.s0 = 0.;
        res.s1 = 0.;
        res.s2 = 1.;
        
        if (abs(y) > cl_plist.s0) res.s1 = -sign(y) * cl_plist.s4);

        float z = (abs(y) - cl_plist.s0) * cl_plist.s4;
        if (z > cl_plist.s2)
        {
            res.s0 = 0;
            res.s1 = 0;
        }
        res.s2 = 1.;
        return normalize(res);

    }

    float3 local_n(float8 cl_plist, int i, float x, float y)
    {
        float3 res=0;
        if (i == 1)
            res = local_n1(cl_plist, x, y);
        if (i == 2)
            res = local_n2(cl_plist, x, y);
        return res;
"""

    def __init__(self, *args, **kwargs):
        u"""
        Creates one prismatic element in a lens stack

        *length*: float [mm]. Lens stack length from first tooth tip to last tooth tip.

        *teeth_n*: float. Number of teeth in the stack.

        *y_t*: float [mm]. Tooth height
        
        *y_g_first*: float [mm]. Distance between first tooth pair.

        *y_g_last*: float [mm]. Distance between last tooth pair.

        *ii*: int. Prism index on the stack starting from 0.
        """
        kwargs = self.__pop_kwargs(**kwargs)
        roe.Plate.__init__(self, *args, **kwargs)
        self.calc_from_stack_params()
        self.center = self.center + np.array([0., self.extra_center_y, 0.])

    def calc_from_stack_params(self):
        if self.y_g_first <= self.y_g_last:
            self.jaws_sign = 1.
        else:
            self.jaws_sign = -1.
        self.half_opening = (self.y_g_first + 
                             self.jaws_sign * self.ii * np.abs(self.y_g_first - self.y_g_last) / (self.teeth_n - 1))
        self.extra_center_y = (self.ii * np.sqrt(self.length ** 2 - (self.y_g_first - self.y_g_last) ** 2) / 
                               (self.teeth_n - 1))
        self.z1max = (np.sqrt(self.length ** 2 - (self.y_g_first - self.y_g_last) ** 2) / (2. * (self.teeth_n - 1)) + 
                       self.jaws_sign * self.y_t * np.abs(self.y_g_first - self.y_g_last) / self.length)
        self.z2max = (np.sqrt(self.length ** 2 - (self.y_g_first - self.y_g_last) ** 2) / (2. * (self.teeth_n - 1)) - 
                       self.jaws_sign * self.y_t * np.abs(self.y_g_first - self.y_g_last) / self.length)

        self.k1 = np.tan(np.pi / 2. - np.arctan2(2. * self.y_t * (self.teeth_n - 1), self.length) + 
                         self.jaws_sign * np.abs(np.arcsin((self.y_g_first - self.y_g_last) / self.length)))
        self.k2 = np.tan(np.pi / 2. - np.arctan2(2. * self.y_t * (self.teeth_n - 1), self.length) - 
                         self.jaws_sign * np.abs(np.arcsin((self.y_g_first - self.y_g_last) / self.length)))

    def __pop_kwargs(self, **kwargs):

        self.length = kwargs.pop('length', 100.)
        self.teeth_n = kwargs.pop('teeth_n', 100)
        self.y_t = kwargs.pop('y_t', 1.)
        self.y_g_first = kwargs.pop('y_g_first', 0.)
        self.y_g_last = kwargs.pop('y_g_last', 1.)
        self.ii = kwargs.pop('ii', 0.)
        
        self.nCRL = 1
        kwargs['pitch'] = np.pi/2
        kwargs['t'] = 0.  
        return kwargs

    def assign_auto_material_kind(self, material):
        material.kind = 'lens'

    def local_z1(self, x, y):
        """Determines the surface of OE at (x, y) position."""
        z = np.zeros_like(y)
        z[np.abs(y) > self.half_opening] = (np.abs(y[np.abs(y) > self.half_opening]) - self.half_opening) * self.k1
        if self.z1max is not None:
            z[z > self.z1max] = self.z1max
        return z

    def local_z2(self, x, y):
        """Determines the surface of OE at (x, y) position."""
        z = np.zeros_like(y)
        z[np.abs(y) > self.half_opening] = (np.abs(y[np.abs(y) > self.half_opening]) - self.half_opening) * self.k2
        if self.z2max is not None:
            z[z > self.z2max] = self.z2max
        return z

    def local_n1(self, x, y):
        a = np.zeros_like(x)
        b = np.zeros_like(x)
        c = np.ones_like(x)

        b[np.abs(y) > self.half_opening] = -np.sign(y[np.abs(y) > self.half_opening]) * self.k1
        z = self.local_z1(x, y)
        b[z >= self.z1max] = 0.

        abc_norm = np.sqrt(a * a + b * b + c * c)
        return [a / abc_norm, b / abc_norm, c / abc_norm]

    def local_n2(self, x, y):
        a = np.zeros_like(x)
        b = np.zeros_like(x)
        c = np.ones_like(x)

        b[np.abs(y) > self.half_opening] = -np.sign(y[np.abs(y) > self.half_opening]) * self.k2
        z = self.local_z2(x, y)
        b[z > self.z2max] = 0.

        abc_norm = np.sqrt(a * a + b * b + c * c)
        return [a / abc_norm, b / abc_norm, c / abc_norm]

    @staticmethod
    def make_stack(L=30., N=30, d=1. , g_first=0., g_last=0., **kwargs):
        center = np.array(kwargs.pop('center', [0, 0, 0]))
        name = kwargs.pop('name', 'Lens') + '_{0:02d}'

        result = []
        for ii in range(N):
            result.append(
                PrismaticLens(
                    name=name.format(ii),
                    center=center,
                    length=L,
                    teeth_n=N,
                    y_t=d,
                    y_g_first=g_first,
                    y_g_last=g_last,
                    ii=ii,
                    **kwargs
                )
            )
        return result

    @staticmethod
    def calc_optimal_params(mat, fdist, en):
        abs_len = 10. / mat.get_absorption_coefficient(en)  # mm
        delta = np.real(1. - mat.get_refractive_index(en))
        
        result = dict()
        result['Focus'] = fdist
        result['SigmaAbs'] = np.sqrt(fdist * delta * abs_len)
        result['Aperture'] = 6. * result['SigmaAbs'] 
        result['y_t'] = 1.64 * result['SigmaAbs']
        result['y_g'] = 1.64 * result['SigmaAbs']
        result['L'] = 2.7 * abs_len

        return result

    @staticmethod
    def calc_y_g(mat, fdist, en, y_t, L):
        delta = np.real(1. - mat.get_refractive_index(en))
        return fdist * delta * L / y_t
