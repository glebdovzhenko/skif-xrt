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
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing

from utils.xrtutils import get_integral_breadth
from utils.various import datafiles
from components import CrystalSiPrecalc, BentLaueParaboloid, BentLaueParaboloidWithBump
from params.sources import ring_kwargs, wiggler_1_5_kwargs
from params.params_1_5 import front_end_distance, front_end_opening, front_end_v_angle, front_end_h_angle, \
    monochromator_distance, monochromator_z_offset, monochromator_x_lim, monochromator_y_lim, \
    exit_slit_distance, exit_slit_opening, filter1_distance, filter2_distance, filter3_distance, filter4_distance, \
    filter5_distance, diamond_filter_thickness, sic_filter_thickness, focusing_distance



# ################################################# SETUP PARAMETERS ###################################################


""" Monochromator """
monochromator_alignment_energy = 30.e3
monochromator_c1_alpha = np.radians(35.3)
monochromator_c1_thickness = 1.8   # meridional
monochromator_c1_thickness_s = .5  # sagittal
monochromator_c2_alpha = np.radians(35.3)
monochromator_c2_thickness = 2.2   # meridional
monochromator_c2_thickness_s = .5  # sagittal


# #################################################### MATERIALS #######################################################


# cr_si_1 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c1_thickness)
# cr_si_2 = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c2_thickness)
cr_si_1 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c1_thickness)
cr_si_2 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c2_thickness)
# cr_si_1 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c1_thickness_s, 
#                            database='/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/components/Si111ref_sag.csv')
# cr_si_2 = CrystalSiPrecalc(hkl=(1, 1, 1), geom='Laue reflection', useTT=True, t=monochromator_c2_thickness_s, 
#                            database='/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/components/Si111ref_sag.csv')
filter_diamond = rm.Material('C', rho=3.5)
filter_si_c = rm.Material(('Si', 'C'), quantities=(1, 1), rho=3.16)
filter_si = rm.Material('Si', rho=2.33)


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

        
        self.FilterBlock = []
        diamond_distances = [filter1_distance, filter1_distance + 1, filter1_distance + 2, 
                             filter2_distance, filter3_distance]
        for ii, dd in enumerate(diamond_distances):
            self.FilterBlock.append(
                roe.Plate(
                    name='Diamond Filter %d' % (ii + 1),
                    bl=self,
                    center=[0, dd, 0],
                    pitch=np.pi/2.,
                    material=filter_diamond,
                    t=diamond_filter_thickness
                )
            )

        sic_distances = [filter4_distance, filter4_distance + 1, filter4_distance + 2, 
                         filter4_distance + 3, filter4_distance + 4, filter5_distance]
        for ii, dd in enumerate(sic_distances):
            self.FilterBlock.append(
                roe.Plate(
                    name='SiC Filter %d' % (ii + 1),
                    bl=self,
                    center=[0, dd, 0],
                    pitch=np.pi/2.,
                    material=filter_si_c,
                    t=sic_filter_thickness
                )
            )

        self.FilterTest = roe.Plate(
                    name='Si Filter Test',
                    bl=self,
                    center=[0, monochromator_distance - 100, 0],
                    pitch=np.pi/2.,  # + np.radians(35.3),
                    material=filter_si,
                    t=1.8 / np.cos(np.radians(35.3))
                ) 

        # self.MonochromatorCr1 = BentLaueParaboloid(
        self.MonochromatorCr1 = BentLaueParaboloidWithBump(
            bl=self,
            name=r'Si[111] Crystal 1',
            center=[0., monochromator_distance, 0.],
            pitch=np.pi / 2.,
            roll=0.,
            yaw=0.,
            alpha=monochromator_c1_alpha,
            material=(cr_si_1,),
            # R=np.inf,
            targetOpenCL='CPU',
            # bendingOrientation='sagittal',
            r_for_refl='y',
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

        self.MonochromatorCr2 = BentLaueParaboloid(
            bl=self,
            name=r'Si[111] Crystal 2',
            center=[0., monochromator_distance, monochromator_z_offset],
            positionRoll=np.pi,
            pitch=0.,
            roll=0.,
            yaw=0.,
            alpha=monochromator_c2_alpha,
            material=(cr_si_2,),
            # R=np.inf,
            targetOpenCL='CPU',
            # bendingOrientation='sagittal',
            r_for_refl='y',
            limPhysY=monochromator_y_lim,
            limOptY=monochromator_y_lim,
            limPhysX=monochromator_x_lim,
            limOptX=monochromator_x_lim,
        )
        self.MonochromatorCr2.ucl = mcl.XRT_CL(r'materials.cl', targetOpenCL='CPU')
        
        self.FocusingMonitor = rscreens.Screen(
            bl=self,
            name=r"Focusing Monitor",
            center=[0, focusing_distance, monochromator_z_offset],
        )

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

        for element in (self.MonochromatorCr1, self.MonochromatorCr2):
            print('#' * 5, element.name, 'RxRyRz', element.pitch, element.roll, element.yaw)

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
        self.MonochromatorCr1.pitch = np.pi / 2 + theta0 + self.MonochromatorCr1.alpha
        self.MonochromatorCr1.center = [
            0.,
            monochromator_distance,
            0.
        ]

        # Crystal 2
        self.MonochromatorCr2.pitch = np.pi / 2 - theta0 + self.MonochromatorCr2.alpha
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
        dd = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'skif15', 'de_to_e_br')

        
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
        dd = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'skif15', 'de_to_e_br')

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
            elif (name == 'C1') or (name == 'C1PA'):
                dist = 1.5 * self.MonochromatorCr1.center[1]
            elif name == 'C1C2':
                dist = 2. * self.Cr1Monitor.center[1]
            elif name == 'C2':
                dist = self.MonochromatorCr2.center[1]
            elif name == 'EM':
                dist = 2. * self.ExitMonitor.center[1]
            elif name == 'ES':
                dist = self.ExitSlit.center[1]
                # elif name in ('F1I', 'F1P', 'F1IT', 'F1PA', 'F2I', 'F2P', 'F2IT', 'F2PA'):
            elif re.match(r'^F[\d]+(I|P|PA|IT)$', name):
                dist = self.MonochromatorCr1.center[1]
            else:
                pass
            
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


def run_process_filters(bl: SKIF15):
    """"""

    beam_source = bl.sources[0].shine()

    beam_ap1 = bl.FrontEnd.propagate(
        beam=beam_source
    )

    bf1g, bf1l, bf1l2 = bl.FilterBlock[0].double_refract(beam=beam_source)
    bf1l2a = raycing.sources.Beam(copyFrom=bf1l2)
    bf1l2a.absorb_intensity(beam_source)

    bf2g, bf2l, bf2l2 = bl.FilterBlock[1].double_refract(beam=bf1g)
    bf2l2a = raycing.sources.Beam(copyFrom=bf2l2)
    bf2l2a.absorb_intensity(bf1g)

    bf3g, bf3l, bf3l2 = bl.FilterBlock[2].double_refract(beam=bf2g)
    bf3l2a = raycing.sources.Beam(copyFrom=bf3l2)
    bf3l2a.absorb_intensity(bf2g)

    bf4g, bf4l, bf4l2 = bl.FilterBlock[3].double_refract(beam=bf3g)
    bf4l2a = raycing.sources.Beam(copyFrom=bf4l2)
    bf4l2a.absorb_intensity(bf3g)

    bf5g, bf5l, bf5l2 = bl.FilterBlock[4].double_refract(beam=bf4g)
    bf5l2a = raycing.sources.Beam(copyFrom=bf5l2)
    bf5l2a.absorb_intensity(bf4g)

    bf6g, bf6l, bf6l2 = bl.FilterBlock[5].double_refract(beam=bf5g)
    bf6l2a = raycing.sources.Beam(copyFrom=bf6l2)
    bf6l2a.absorb_intensity(bf5g)

    bf7g, bf7l, bf7l2 = bl.FilterBlock[6].double_refract(beam=bf6g)
    bf7l2a = raycing.sources.Beam(copyFrom=bf7l2)
    bf7l2a.absorb_intensity(bf6g)

    bf8g, bf8l, bf8l2 = bl.FilterBlock[7].double_refract(beam=bf7g)
    bf8l2a = raycing.sources.Beam(copyFrom=bf8l2)
    bf8l2a.absorb_intensity(bf7g)

    bf9g, bf9l, bf9l2 = bl.FilterBlock[8].double_refract(beam=bf8g)
    bf9l2a = raycing.sources.Beam(copyFrom=bf9l2)
    bf9l2a.absorb_intensity(bf8g)

    bf10g, bf10l, bf10l2 = bl.FilterBlock[9].double_refract(beam=bf9g)
    bf10l2a = raycing.sources.Beam(copyFrom=bf10l2)
    bf10l2a.absorb_intensity(bf9g)

    bf11g, bf11l, bf11l2 = bl.FilterBlock[10].double_refract(beam=bf10g)
    bf11l2a = raycing.sources.Beam(copyFrom=bf11l2)
    bf11l2a.absorb_intensity(bf10g)

    # beam_mono_c1_global, beam_mono_c1_local = bl.MonochromatorCr1.reflect(
    #     beam=beam_filter_global[ff_name + 'Global'], returnLocalAbsorbed=True
    # ) 
    # beam_mono_c1_local.absorb_intensity(beam_filter_global[ff_name + 'Global'])

    beam_mono_c1_global, beam_mono_c1_local, beam_mono_c1_local2 = bl.FilterTest.double_refract(
        beam=bf11g
    )
    beam_mono_c1_local2a = raycing.sources.Beam(copyFrom=beam_mono_c1_local2)
    beam_mono_c1_local2a.absorb_intensity(bf11g)

    bl.prepare_flow()

    return {
        'BeamSourceGlobal': beam_source,
        'BeamAperture1Local': beam_ap1,
        'BeamMonoC1Local2a': beam_mono_c1_local2a,

        'BeamFilter1Local1': bf1l,
        'BeamFilter2Local1': bf2l,
        'BeamFilter3Local1': bf3l,
        'BeamFilter4Local1': bf4l,
        'BeamFilter5Local1': bf5l,
        'BeamFilter6Local1': bf6l,
        'BeamFilter7Local1': bf7l,
        'BeamFilter8Local1': bf8l,
        'BeamFilter9Local1': bf9l,
        'BeamFilter10Local1': bf10l,
        'BeamFilter11Local1': bf11l,

        'BeamFilter1Local2': bf1l2,
        'BeamFilter2Local2': bf2l2,
        'BeamFilter3Local2': bf3l2,
        'BeamFilter4Local2': bf4l2,
        'BeamFilter5Local2': bf5l2,
        'BeamFilter6Local2': bf6l2,
        'BeamFilter7Local2': bf7l2,
        'BeamFilter8Local2': bf8l2,
        'BeamFilter9Local2': bf9l2,
        'BeamFilter10Local2': bf10l2,
        'BeamFilter11Local2': bf11l2,

        'BeamFilter1Local2a': bf1l2a,
        'BeamFilter2Local2a': bf2l2a,
        'BeamFilter3Local2a': bf3l2a,
        'BeamFilter4Local2a': bf4l2a,
        'BeamFilter5Local2a': bf5l2a,
        'BeamFilter6Local2a': bf6l2a,
        'BeamFilter7Local2a': bf7l2a,
        'BeamFilter8Local2a': bf8l2a,
        'BeamFilter9Local2a': bf9l2a,
        'BeamFilter10Local2a': bf10l2a,
        'BeamFilter11Local2a': bf11l2a,
    }


def run_process(bl: SKIF15):
    """"""

    beam_source = bl.sources[0].shine()

    beam_ap1 = bl.FrontEnd.propagate(
        beam=beam_source
    )

    beam_mono_c1_global, beam_mono_c1_local = bl.MonochromatorCr1.reflect(
        beam=beam_source #, returnLocalAbsorbed=True
    )

    beam_mon1 = bl.Cr1Monitor.expose(
        beam=beam_mono_c1_global
    )

    beam_mono_c2_global, beam_mono_c2_local = bl.MonochromatorCr2.reflect(
        beam=beam_mono_c1_global #, returnLocalAbsorbed=True
    )
    
    beam_monf = bl.FocusingMonitor.expose(
        beam=beam_mono_c2_global
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
        # 'BeamFilter1Global': beam_filter1_global, 
        # 'BeamFilter1Local1': beam_filter1_local1, 
        # 'BeamFilter1Local2': beam_filter1_local2,
        # 'BeamFilter1Local2a': beam_filter1_local2a, 
        # 'BeamFilter2Global': beam_filter2_global, 
        # 'BeamFilter2Local1': beam_filter2_local1, 
        # 'BeamFilter2Local2': beam_filter2_local2,
        # 'BeamFilter2Local2a': beam_filter2_local2a,
        'BeamMonoC1Local': beam_mono_c1_local,
        'BeamMonoC1Global': beam_mono_c1_global,
        'BeamMonitor1Local': beam_mon1,
        'BeamMonoC2Local': beam_mono_c2_local,
        'BeamMonoC2Global': beam_mono_c2_global,
        'BeamAperture2Local': beam_ap2,
        'BeamMonitor2Local': beam_mon2,
        'BeamFocusingMonitorLocal': beam_monf,
    }


rrun.run_process = run_process

