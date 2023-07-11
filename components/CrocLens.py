import numpy as np

import xrt.backends.raycing.oes as roe


class CrocLens(roe.Plate):
    r"""
    """

    hiddenMethods = roe.Plate.hiddenMethods + ['double_refract']
    cl_plist = ("z1max", "z2max", "t_angle", "t_offset_angle", "aperture")
    cl_local_z = """
    float local_z1(float8 cl_plist, float x, float y)
    {
        float res;
        res = (abs(y) - 0.5 * cl_plist.s4) * tan((cl_plist.s2 + cl_plist.s3) / 2);
        if (abs(y) < 0.5 * cl_plist.s4) res = 0.;
        if  (res > cl_plist.s0) res = cl_plist.s0;
        return res;
    }

    float local_z2(float8 cl_plist, float x, float y)
    {
        float res;
        res = (abs(y) - 0.5 * cl_plist.s4) * tan((cl_plist.s2 - cl_plist.s3) / 2);
        if (abs(y) < 0.5 * cl_plist.s4) res = 0.;
        if  (res > cl_plist.s1) res = cl_plist.s1;
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
    }"""
    cl_local_n = """
    float3 local_n1(float8 cl_plist, float x, float y)
    {
        float3 res;
        res.s0 = 0.;
        res.s1 = 0.;
        res.s2 = 1.;
        
        if (abs(y) > cl_plist.s4) res.s1 = -sign(y) * tan((cl_plist.s2 + cl_plist.s3) / 2);

        float z = (abs(y) - 0.5 * cl_plist.s4) * tan((cl_plist.s2 + cl_plist.s3) / 2);
        if (z > cl_plist.s0)
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
        
        if (abs(y) > cl_plist.s4) res.s1 = -sign(y) * tan((cl_plist.s2 - cl_plist.s3) / 2);

        float z = (abs(y) - 0.5 * cl_plist.s4) * tan((cl_plist.s2 - cl_plist.s3) / 2);
        if (z > cl_plist.s1)
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
    }"""


    def __init__(self, *args, **kwargs):
        u"""
        *aperture*: float 
            (see image above), [mm], default is 2.

        *t_angle*: float 
            (see image above) [rad], default is π/3 for equilateral.

        *t_offset_angle*: float
            [rad] angle between jaws, default 0
        
        *t*: float
            plate thickness should be 0 and is ignored.

        *pitch*: float
            the default value is set to π/2, i.e. to normal incidence.

        *z1max*: float
            If given, limits the *z* coordinate; the object becomes then a
            plate of the thickness *zmax* + *t* with a paraboloid hole at the
            origin.

        *z2max*: float
            Same as z1max, but from the backside
        """
        kwargs = self.__pop_kwargs(**kwargs)
        roe.Plate.__init__(self, *args, **kwargs)

    def __pop_kwargs(self, **kwargs):
        self.aperture = kwargs.pop('aperture', 2.)
        self.t_angle = kwargs.pop('t_angle', np.pi / 3)
        self.t_offset_angle = kwargs.pop('t_offset_angle', 0)
        self.z1max = kwargs.pop('z1max', None)
        self.z2max = kwargs.pop('z2max', None)

        self.nCRL = kwargs.pop('nCRL', 1)
        kwargs['pitch'] = kwargs.get('pitch', np.pi/2)
        kwargs['t'] = 0.  # kwargs.get('t', 0.)
        return kwargs

    def assign_auto_material_kind(self, material):
        material.kind = 'lens'

    def local_z1(self, x, y):
        """Determines the surface of OE at (x, y) position."""
        z = np.zeros_like(y)
        z[np.abs(y) > self.aperture / 2] = (np.abs(y[np.abs(y) > self.aperture / 2]) - self.aperture / 2) * np.tan((self.t_angle + self.t_offset_angle) / 2)
        if self.z1max is not None:
            z[z > self.z1max] = self.z1max
        return z

    def local_z2(self, x, y):
        """Determines the surface of OE at (x, y) position."""
        z = np.zeros_like(y)
        z[np.abs(y) > self.aperture / 2] = (np.abs(y[np.abs(y) > self.aperture / 2]) - self.aperture / 2) * np.tan((self.t_angle - self.t_offset_angle) / 2)
        if self.z2max is not None:
            z[z > self.z2max] = self.z2max
        return z

    def local_n1(self, x, y):
        a = np.zeros_like(x)
        b = np.zeros_like(x)
        c = np.ones_like(x)

        b[np.abs(y) > self.aperture / 2] = -np.sign(y[np.abs(y) > self.aperture / 2]) * np.tan((self.t_angle + self.t_offset_angle) / 2)
        z = (np.abs(y) - self.aperture / 2) * np.tan((self.t_angle + self.t_offset_angle) / 2)
        b[z > self.z1max] = 0.

        abc_norm = np.sqrt(a * a + b * b + c * c)
        return [a / abc_norm, b / abc_norm, c / abc_norm]

    def local_n2(self, x, y):
        a = np.zeros_like(x)
        b = np.zeros_like(x)
        c = np.ones_like(x)

        b[np.abs(y) > self.aperture / 2] = -np.sign(y[np.abs(y) > self.aperture / 2]) * np.tan((self.t_angle - self.t_offset_angle) / 2)
        z = (np.abs(y) - self.aperture / 2) * np.tan((self.t_angle - self.t_offset_angle) / 2)
        b[z > self.z2max] = 0.

        abc_norm = np.sqrt(a * a + b * b + c * c)
        return [a / abc_norm, b / abc_norm, c / abc_norm]

    @staticmethod
    def make_stack(L=30., N=30, d=None, theta=None, g_left=0., g_right=0., **kwargs):
        if d is None and theta is None:
            theta = np.pi/3
            d = L * np.tan(theta) / (2. * N)
        elif theta is None and d is not None:
            theta = np.arctan(2. * N * d / L)
        elif d is None and theta is not None:
            d = L * np.tan(theta) / (2. * N)
        else:
            raise ValueError('make_stack accepts d OR theta, the other (or both for default) should be None')

        alpha = np.arcsin(.5 * (g_right - g_left) / L)
        center = np.array(kwargs.pop('center', [0, 0, 0]))
        name = kwargs.pop('name', 'Lens') + '_{0:02d}'

        result = []
        for ii in range(N):
            result.append(
                CrocLens(
                    name=name.format(ii),
                    center=center + np.array([0., ii * L * np.cos(alpha / 2) / (N - 1), 0.]),
                    pitch=np.pi/2.,
                    z2max=L * np.cos(alpha / 2) / N - d * np.sin(np.pi/2 - theta - alpha / 2.) / np.sin(theta),
                    z1max=d * np.sin(np.pi/2 - theta - alpha / 2.) / np.sin(theta),
                    aperture=2. * (g_left + (g_right-g_left) * ii / (N - 1)),
                    t_angle=np.pi - 2. * theta,
                    t_offset_angle=-alpha,
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
