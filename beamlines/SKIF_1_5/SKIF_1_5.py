import numpy as np
from typing import List
import pickle
import os
import re

import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.myopencl as mcl
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing

from utils.xrtutils import get_integral_breadth
from utils.various import datafiles
from components import BentLaueCylinder
from params.sources import ring_kwargs, wiggler_1_5_kwargs
from params.params_1_5 import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle, \
    monochromator_distance, monochromator_z_offset, monochromator_x_lim, monochromator_y_lim, \
    exit_slit_distance, exit_slit_opening



# ################################################# SETUP PARAMETERS ###################################################


""" Monochromator """
monochromator_alignment_energy = 30.e3
monochromator_c1_alpha = np.radians(21.9)
monochromator_c1_thickness = 1.2
monochromator_c2_alpha = np.radians(21.9)
monochromator_c2_thickness = 1.2


# #################################################### MATERIALS #######################################################


cr_si_1 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c1_thickness)
cr_si_2 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c2_thickness)


# #################################################### BEAMLINE ########################################################


class SKIF15(raycing.BeamLine):
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
            name=r"Crystal 1-2 Monitor",
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
            R=np.inf,
            targetOpenCL='CPU',
            limPhysY=monochromator_y_lim,
            limOptY=monochromator_y_lim,
            limPhysX=monochromator_x_lim,
            limOptX=monochromator_x_lim,
        )
        self.MonochromatorCr2.ucl = mcl.XRT_CL(r'materials.cl', targetOpenCL='CPU')

        self.ExitMonitor = rscreens.Screen(
            bl=self,
            name=r"Exit Monitor",
            center=[0, exit_slit_distance - 10, monochromator_z_offset],
        )

        self.ExitSlit = rapts.RectangularAperture(
            bl=self,
            name=r"Exit Slit",
            center=[0, exit_slit_distance, monochromator_z_offset],
            opening=exit_slit_opening
        )
    
    def dumps(self):
        result = ''

        result += 'E: %s\n' % str(self.alignE)
        for (name, obj) in zip(('C1', 'C2'), (self.MonochromatorCr1, self.MonochromatorCr2)):
            result += '%s\n' % name
            result += 'XYZ: %s\n' % str(obj.center)
            result += 'PRY: %s\n' % str([obj.pitch, obj.roll, obj.yaw])
            result += 'TAR: %s\n' % str([obj.material[0].t, np.degrees(obj.alpha), obj.R])

        return result

    def print_positions(self):
        print('#' * 20, self.name, '#' * 20)

        for element in (self.SuperCWiggler, self.FrontEnd,
                        self.MonochromatorCr1, self.Cr1Monitor, self.MonochromatorCr2,
                        self.ExitMonitor, self.ExitSlit):
            print('#' * 5, element.name, 'at', element.center)

        print('#' * (42 + len(self.name)))


    def align_energy(self, en, d_en, mono=False):
        # changing energy for the beamline / source
        self.alignE = en
        if not mono:
            self.SuperCWiggler.eMin = en * (1. - d_en)
            self.SuperCWiggler.eMax = en * (1. + d_en)
        else:
            self.SuperCWiggler.eMin = en - 1.
            self.SuperCWiggler.eMax = en + 1.

        # Diffraction angle for the DCM
        theta0 = np.arcsin(rm.ch / (2 * self.MonochromatorCr1.material[0].d * en))

        # Setting up DCM orientations / positions
        # Crystal 1
        self.MonochromatorCr1.pitch = np.pi / 2 + theta0 + monochromator_c1_alpha
        self.MonochromatorCr1.set_alpha(monochromator_c1_alpha)
        self.MonochromatorCr1.center = [
            0.,
            monochromator_distance,
            0.
        ]

        # Crystal 2
        self.MonochromatorCr2.pitch = np.pi / 2 - theta0 - monochromator_c2_alpha
        self.MonochromatorCr2.set_alpha(-monochromator_c2_alpha)
        self.MonochromatorCr2.center = [
            0.,
            monochromator_distance + monochromator_z_offset / np.tan(2. * theta0),
            monochromator_z_offset
        ]

        # between-crystals monitor
        self.Cr1Monitor.center = [
            0.,
            monochromator_distance + .5 * monochromator_z_offset / np.tan(2. * theta0),
            .5 * monochromator_z_offset
        ]

        self.print_positions()

    @staticmethod
    def get_de_over_e(r, en):
        """
        Expected dE/E of the monochromator crystal calculated from its bending radius.
        For meridional bend with focus behind the crystal.
        Can be used as wiggler energy limits, plot limits, etc.
        """
        dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/de_to_e_br'

        
        rs, de = [], []
        for metadata in datafiles(dd):
            if metadata['name'] != 'C1C2' or metadata['axes'] != 'XXpr':
                continue

            rs.append(metadata['r1'])

            with open(os.path.join(dd, metadata['file']), 'rb') as f:
                data = pickle.load(f)
                de.append(get_integral_breadth(data, axis='e'))

        rs, de = np.array(rs), np.array(de)
        ii = np.argsort(rs)
        rs, de = rs[ii], de[ii]

        return (1.5 + 2.5 * np.abs(en - 30.e3) / 120.e3) * np.interp(r, 1e3 * rs, de) / 30.e3

    @staticmethod
    def get_dzpr(r):
        dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/de_to_e_br'

        rs, dzpr = [], []
        for metadata in datafiles(dd):
            if metadata['name'] != 'C1C2' or metadata['axes'] != 'ZZpr':
                continue

            rs.append(metadata['r1'])

            with open(os.path.join(dd, metadata['file']), 'rb') as f:
                data = pickle.load(f)
                dzpr.append(get_integral_breadth(data, axis='y'))

        rs, dzpr = np.array(rs), np.array(dzpr)
        ii = np.argsort(rs)
        rs, dzpr = rs[ii], dzpr[ii]

        return np.interp(r, 1e3 * rs, dzpr)


    def set_plot_limits(self, plts: List):
        for plot in plts:
            # adjusting energy limits and offset
            plot.caxis.offset = .5 * (self.SuperCWiggler.eMin + self.SuperCWiggler.eMax)
            plot.caxis.limits = [self.SuperCWiggler.eMin, self.SuperCWiggler.eMax]

            name, axes, = plot.title.split('-')
            axes = re.match(r'^(X|Xpr|Z|Zpr)(X|Xpr|Z|Zpr)$', axes)
            axes = [axes[1], axes[2]]
            
            if name == 'FE':
                dist = self.FrontEnd.center[1]
            elif name == 'C1':
                dist = self.MonochromatorCr1.center[1]
            elif name == 'C1C2':
                dist = 2. * self.Cr1Monitor.center[1]
            elif name == 'C2':
                dist = self.MonochromatorCr2.center[1]
            elif name == 'EM':
                dist = 2. * self.ExitMonitor.center[1]
            elif name == 'ES':
                dist = self.ExitSlit.center[1]
            else:
                raise ValueError('Unknown plot type: %s' % plot.title)

            for ax, ax_name in zip((plot.xaxis, plot.yaxis), axes):

                if (name == 'C1C2') and (ax_name == 'Zpr'):
                    ax.offset = self.Cr1Monitor.center[2] / \
                    (self.Cr1Monitor.center[1] - self.MonochromatorCr1.center[1])
                else:
                    ax.offset = 0.

                if ax_name == 'X':
                    ax.limits = [-dist * np.tan(front_end_h_angle / 2.), dist * np.tan(front_end_h_angle / 2.)]
                elif ax_name == 'Z':
                    ax.limits = [-dist * np.tan(front_end_v_angle / 2.), dist * np.tan(front_end_v_angle / 2.)]
                elif ax_name == 'Zpr':
                    dzpr = self.get_dzpr(np.min([self.MonochromatorCr1.R, self.MonochromatorCr2.R]))
                    if name in ('EM', 'ES'):
                        ax.limits = [-np.tan(front_end_v_angle / 2.), np.tan(front_end_v_angle / 2.)]
                    elif name == 'C1C2':
                        ax.limits = [ax.offset - dzpr, ax.offset + dzpr]
                elif ax_name == 'Xpr':
                    ax.limits = [-np.tan(front_end_h_angle / 2.), np.tan(front_end_h_angle / 2.)]


# ################################################# BEAM TOPOLOGY ######################################################


def run_process(bl: SKIF15):
    """"""
    # global show

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

    beam_mon2 = bl.ExitMonitor.expose(
        beam=beam_mono_c2_global
    )

    beam_ap2 = bl.ExitSlit.propagate(
        beam=beam_mono_c2_global
    )

    bl.prepare_flow()

    return {
        'BeamSourceGlobal': beam_source,
        'BeamAperture1Local': beam_ap1,
        'BeamMonoC1Local': beam_mono_c1_local,
        'BeamMonoC1Global': beam_mono_c1_global,
        'BeamMonitor1Local': beam_mon1,
        'BeamMonoC2Local': beam_mono_c2_local,
        'BeamMonoC2Global': beam_mono_c2_global,
        'BeamAperture2Local': beam_ap2,
        'BeamMonitor2Local': beam_mon2,
    }


rrun.run_process = run_process

