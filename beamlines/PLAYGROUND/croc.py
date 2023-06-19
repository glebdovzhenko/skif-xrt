from os import wait
from typing import List
import numpy as np
import os
import matplotlib
matplotlib.use('agg')

import xrt.backends.raycing as raycing
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.run as rrun
import xrt.runner as xrtrun
import xrt.plotter as xrtplot
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing as raycing

from params.sources import ring_kwargs, wiggler_nstu_scw_kwargs
from params.params_nstu_scw import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle 


mBeryllium = rm.Material('Be', rho=1.848, kind='lens')
first_lens_distance = 17000.
focus_screen_dist = 20000.

# Lens params, nomenclature by W. Jark: A simple x-ray monochromator based on an alligator lens:
croc_L = 30.  # mm, full jaw length
croc_N = 30  # number of teeth on a jaw
croc_d = 1.7  # mm, tooth height
croc_Theta = np.arctan(2. * croc_d / (croc_L / croc_N))
croc_g_left = 0.  # mm half of the left (closer to source) opening of the lens
croc_g_right = 2.  # mm half of the right (further from source) opening of the lens


class CrocLens(roe.Plate):
    r"""
    Alligator-type CRL. Each tooth is defined by aperture (distance between tooth tips) and angle of the 
    triangle
    \      /
     \    /
      \  /
       \/
             /\ 
             | aperture
      angle  \/
       /\
      /  \
     /    \
    /      \
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
        print(alpha)
        for ii in range(N):
            result.append(
                CrocLens(
                    name=name.format(ii),
                    center=center + np.array([0., ii * L * np.cos(alpha / 2) / (N - 1), 0.]),
                    pitch=np.pi/2.,
                    z2max=L * np.cos(alpha / 2) / N - d * np.sin(np.pi/2 - theta - alpha / 2.) / np.sin(theta),
                    z1max=d * np.sin(np.pi/2 - theta - alpha / 2.) / np.sin(theta),
                    aperture=g_left + (g_right-g_left) * ii / (N - 1),
                    t_angle=np.pi - 2. * theta,
                    t_offset_angle=-alpha,
                    **kwargs
                )
            )
        return result


class CrocBL(raycing.BeamLine):
    def __init__(self, azimuth=0, height=0, alignE='auto'):
        super().__init__(azimuth, height, alignE)
        
        self.name = 'CrocBL' 

        self.SuperCWiggler = rsources.Wiggler(
            name=r"Superconducting Wiggler",
            bl=self,
            center=[0, 0, 0],
            eMin=100,
            eMax=100000,
            xPrimeMax=front_end_h_angle * .505e3,
            zPrimeMax=front_end_v_angle * .505e3,
            **ring_kwargs,
            **wiggler_nstu_scw_kwargs
        )

        self.FrontEnd = rapts.RectangularAperture(
            bl=self,
            name=r"Front End Slit",
            center=[0, front_end_distance, 0],
            opening=front_end_opening
        )

        self.LensStack = CrocLens.make_stack(
            L=30., N=croc_N, theta=np.pi/3., g_left=4., g_right=0.,
            bl=self, 
            center=[0., first_lens_distance, 0],
            material=mBeryllium,
            limPhysX=[-20, 20], 
            limPhysY=[-5, 5], 
        )
        
        self.ExitMonitor = rscreens.Screen(
            bl=self,
            name=r"Exit Monitor",
            center=[0, focus_screen_dist, 0],
        )


def run_process(bl: CrocBL):

    beam_source = bl.sources[0].shine()

    beam_ap1 = bl.FrontEnd.propagate(
        beam=beam_source
    )
    
    outDict = {
        'BeamSourceGlobal': beam_source,
        'BeamAperture1Local': beam_ap1,
    }

    beamIn = beam_source
    for ilens, lens in enumerate(bl.LensStack):
        lglobal, llocal1, llocal2 = lens.double_refract(beamIn, needLocal=True)
        strl = '_{0:02d}'.format(ilens)
        outDict['BeamLensGlobal'+strl] = lglobal
        outDict['BeamLensLocal1'+strl] = llocal1
        outDict['BeamLensLocal2'+strl] = llocal2
        
        llocal2a = raycing.sources.Beam(copyFrom=llocal2)
        llocal2a.absorb_intensity(beamIn)
        outDict['BeamLensLocal2a'+strl] = llocal2a
        beamIn = lglobal

    outDict['BeamMonitor1'] = bl.ExitMonitor.expose(lglobal)

    bl.prepare_flow()
    return outDict

plots = [
    # Front-end
    xrtplot.XYCPlot(
        beam='BeamAperture1Local',
        title='FE-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='BeamMonitor1',
        title='EM-XZ',
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$z$', unit='mm', data=raycing.get_z),
        aspect='auto'),
] + [
    xrtplot.XYCPlot(
        beam='BeamLensLocal1_{0:02d}'.format(ii),
        title='Lens_{0:02d}-XZ'.format(ii),
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$y$', unit='mm', data=raycing.get_y),
        aspect='auto') for ii in range(croc_N)
] + [
    xrtplot.XYCPlot(
        beam='BeamLensLocal2a_{0:02d}'.format(ii),
        title='LensAbs_{0:02d}-XZ'.format(ii),
        xaxis=xrtplot.XYCAxis(label=r'$x$', unit='mm', data=raycing.get_x),
        yaxis=xrtplot.XYCAxis(label=r'$y$', unit='mm', data=raycing.get_y),
        aspect='auto',
        fluxKind='power') for ii in range(croc_N)
]


def onept(plts: List, bl: CrocBL):
    # bl.SuperCWiggler.eMin = 10000
    # bl.SuperCWiggler.eMax = 15000
    for plot in plts:
        plot.xaxis.limits = None
        plot.yaxis.limits = [-2.5, 2.5]
        plot.caxis.limits = None
        plot.saveName = os.path.join(os.getenv('BASE_DIR'), 'datasets', plot.title + '.png')
    yield


if __name__ == '__main__':
    scan = onept
    show = True
    repeats = 2

    beamline = CrocBL()
    rrun.run_process = run_process
    if show:
        beamline.glow(
            scale=[1e3, 1e4, 1e4],
            # centerAt=r'Si[111] Crystal 1',
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
