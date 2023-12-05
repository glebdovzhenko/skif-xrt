import os
import pickle
import shutil
from typing import Dict, List
import matplotlib
import numpy as np
import git
import csv
import re

import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from NSTU_SCW import FL, NSTU_SCW, mGlassyCarbon
from components import PrismaticLens
from params.params_nstu_scw import (
    croc_crl_L,
    croc_crl_distance,
    croc_crl_y_t,
    croc_crl_N,
    diamond_filter_N,
    sic_filter_N,
    front_end_opening,
    monochromator_x_lim,
    monochromator_y_lim,
    filter_size_x,
    filter_size_z,
    sic_filter_N,
)
from utils.xrtutils import get_integral_breadth, get_line_kb, get_minmax, pickle_to_table

matplotlib.use('agg')


plots = []
x_kwds = {r'label': r'$x$', r'unit': r'mm', r'data': raycing.get_x}
y_kwds = {r'label': r'$y$', r'unit': r'mm', r'data': raycing.get_y}
z_kwds = {r'label': r'$z$', r'unit': r'mm', r'data': raycing.get_z}
xpr_kwds = {r'label': r'$x^{\prime}$',
            r'unit': r'', r'data': raycing.get_xprime}
zpr_kwds = {r'label': r'$z^{\prime}$',
            r'unit': r'', r'data': raycing.get_zprime}

# optical elements
for beam, t1 in zip(
        ('BeamAperture1Local', 'BeamMonoC1Local', 'BeamMonitor1Local',
         'BeamMonoC2Local', 'BeamMonitor2Local'),
        ('FE', 'C1', 'C1C2', 'C2', 'FM')):
    if t1 not in ('C1', 'C2'):
        params = zip(('XZ', 'XXpr', 'ZZpr'), (x_kwds, x_kwds,
                     z_kwds), (z_kwds, xpr_kwds, zpr_kwds))
    else:
        params = zip(('XY', 'XXpr'), (x_kwds, x_kwds), (y_kwds, xpr_kwds))

    for t2, xkw, ykw in params:
        plots.append(xrtplot.XYCPlot(beam=beam,
                                     title='-'.join((t1, t2)),
                                     xaxis=xrtplot.XYCAxis(**xkw),
                                     yaxis=xrtplot.XYCAxis(**ykw),
                                     aspect='auto'))
# # filter absorption plots
# for beam in [
#     'BeamFilterCLocal2a_{0:02d}'.format(ii) for ii in range(diamond_filter_N)
# ] + [
#     'BeamFilterSiCLocal2a_{0:02d}'.format(ii) for ii in range(sic_filter_N)
# ]:
#     t1 = beam.replace('BeamFilter', '').replace('Local2a_', '')
#     t2 = 'XZ'
#     plots.append(xrtplot.XYCPlot(beam=beam,
#                                  title='-'.join((t1, t2)),
#                                  xaxis=xrtplot.XYCAxis(
#                                      limits=[-filter_size_x/2, filter_size_x/2], **x_kwds),
#                                  yaxis=xrtplot.XYCAxis(
#                                      limits=[-filter_size_z/2, filter_size_z/2], **y_kwds),
#                                  fluxKind='power',
#                                  aspect='auto'))
# # CRL absorption plots
# for beam in [
#     'BeamLensLocal2a_{0:02d}'.format(ii) for ii in range(croc_crl_N)
# ]:
#     t1 = beam.replace('BeamLensLocal2a_', 'CRL')
#     t2 = 'XZ'
#     plots.append(xrtplot.XYCPlot(beam=beam,
#                                  title='-'.join((t1, t2)),
#                                  xaxis=xrtplot.XYCAxis(
#                                      limits=[-3 * filter_size_x / 4, 3 * filter_size_x / 4], **x_kwds),
#                                  yaxis=xrtplot.XYCAxis(
#                                      limits=[-filter_size_z/2, filter_size_z/2], **y_kwds),
#                                  fluxKind='power',
#                                  aspect='auto'
#                                  ))


def check_repo(md: Dict):
    r = git.Repo(os.getenv('BASE_DIR'))
    assert not r.is_dirty()
    assert r.head.ref == r.heads.rtr
    md['commit'] = r.head.commit.name_rev.replace(' rtr', '')
    return md


def absorbed_power(bl: NSTU_SCW, plts: List):
    subdir = os.path.join(os.getenv('BASE_DIR', ''), 'datasets', 'nstu-scw-2')
    scan_name = 'absorbed_power_lens_w_slits'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    bl.align_source(50050, 999. / 1001.)
    bl.align_crl(croc_crl_L, croc_crl_N, croc_crl_y_t, .12, 0.)
    bl.align_crl_mask(100., 6. * PrismaticLens.calc_sigma_abs(mGlassyCarbon, 14000, 90e3))
    bl.align_mono(50e3, np.inf, np.inf, np.inf, np.inf)

    for plot in plts:
        if re.match(r'(C|SiC|CRL)[\d]+-XZ', plot.title):
            print(plot.title)
            plot.saveName = os.path.join(
                subdir, scan_name, '%s.png' % (plot.title, ))
            plot.persistentName = plot.saveName.replace('.png', '.pickle')

    metadata = check_repo(bl._metadata)
    with open(os.path.join(subdir, scan_name, 'md.csv'), 'w') as ff:
        ff.write('\n'.join('%s,%s' % (k, str(val))
                 for k, val in metadata.items()))
    yield

    # FILTERS
    # for plot in plts:
    #     if plot.persistentName is not None:
    #         with open(plot.persistentName, 'rb') as f:
    #             f = pickle.load(f)
    #             np.savetxt(
    #                 plot.persistentName.replace('.pickle', '.txt'),
    #                 pickle_to_table(f),
    #                 delimiter=' ',
    #                 header="""\"x (mm)\"	\"y (mm)\"	\"Absorbed Power Density (W/mm<sup>2</sup>)\""""
    #             )
    # LENS
    tooth_ids, tooth_ps, tooth_ps_density = [], [], []
    for plot in plts:
        if plot.persistentName is not None:
            with open(plot.persistentName, 'rb') as f:
                f = pickle.load(f)
                tooth_ids.append(int(
                    os.path.basename(plot.persistentName)
                    .replace('CRL', '').replace('-XZ.pickle', '')
                ))
                tooth_ps.append(f.power)
                tooth_ps_density.append(
                    (np.max(f.total2D) / f.nRaysAll) / ((np.mean(f.xbinEdges[1:] - f.xbinEdges[:-1]) * (
                        np.mean(f.ybinEdges[1:] - f.ybinEdges[:-1]))))
                )
    tooth_ps = np.array([tooth_ids, tooth_ps, tooth_ps_density]).T
    tooth_ps = tooth_ps[tooth_ps[:, 0].argsort()]
    tooth_ps[:, 0] = 2. * tooth_ps[:, 0]

    np.savetxt(
        os.path.join(subdir, scan_name, 'lens_power.txt'),
        tooth_ps[:,[0, 1]],
        delimiter=' ',
        header="""\"Y (mm)\" \"Total Power (W)\" """,
    )
    np.savetxt(
        os.path.join(subdir, scan_name, 'lens_power_density.txt'),
        tooth_ps[:,[0, 2]],
        delimiter=' ',
        header="""\"Y (mm)\" \"Max Power Density (W/mm<sup>2</sup>)\"""",
    )

    print('TOTAL CRL ABSORBED POWER %.03f W' % np.sum(tooth_ps[:, 1]))


# @FL.gnrtr(45e3, 67e3, 30)
def onept(bl: NSTU_SCW, plts: List):
    subdir = os.path.join(os.getenv('BASE_DIR', ''), 'datasets', 'nstu-scw-2')
    scan_name = 'exit_w_slits'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 30.e3
    if np.isclose(en, 30e3):
        r1, r2 = -2.04e3, -2.04e3   # 30 keV
        g_f = 1.25                  # 30 keV
        d_en = 5e-3
        six_sigma = 2.1
    elif np.isclose(en, 50e3):
        r1, r2 = -1.22e3, -1.22e3   # 50 keV
        g_f = 0.45                  # 50 keV
        d_en = 5e-3
        six_sigma = 1.5
    elif np.isclose(en, 70e3):
        r1, r2 = -.870e3, -.870e3   # 70 keV
        g_f = .21                   # 70 keV
        d_en = 1e-2
        six_sigma = 1.1
    elif np.isclose(en, 90e3):
        r1, r2 = -.675e3, -.675e3   # 90 keV
        g_f = 0.12                  # 90 keV
        d_en = 3e-2
        six_sigma = 0.9
    else:
        raise ValueError('En is not in [30, 50, 70, 90] keV')

    bl.align_source(en, d_en)
    bl.align_crl(croc_crl_L, croc_crl_N, croc_crl_y_t, g_f, 0.)
    bl.align_crl_mask(100., six_sigma)
    bl.align_mono(en, r1, -6. * r1, r2, -6 * r2)

    for plot in plts:
        plot.saveName = os.path.join(
            subdir,
            scan_name,
            '%s.png' % (plot.title)
        )
        plot.persistentName = plot.saveName.replace('.png', '.pickle')
        if 'FM-XZ' in plot.title:
            plot.xaxis.limits = [-.2, .2]
            plot.yaxis.limits = [-.025, .025]

    metadata = check_repo(bl._metadata)
    with open(os.path.join(subdir, scan_name, 'md.csv'), 'w') as ff:
        ff.write('\n'.join('%s,%s' % (k, str(val))
                 for k, val in metadata.items()))

    yield


def scan_mask_opening(bl: NSTU_SCW, plts: List):
    subdir = os.path.join(os.getenv('BASE_DIR', ''), 'datasets', 'nstu-scw-2')
    scan_name = 'scan_mask_opening_x_50'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 50.e3
    if np.isclose(en, 30e3):
        r1, r2 = -2.04e3, -2.04e3  # 30 keV
        g_f = 1.076                # 30 keV
        d_en = 5e-3
    elif np.isclose(en, 50e3):
        r1, r2 = -1.22e3, -1.22e3  # 50 keV
        g_f = 0.390                # 50 keV
        d_en = 5e-3
    elif np.isclose(en, 70e3):
        r1, r2 = -.870e3, -.870e3  # 70 keV
        g_f = .191                 # 70 keV
        d_en = 1e-2
    elif np.isclose(en, 90e3):
        r1, r2 = -.675e3, -.675e3  # 90 keV
        g_f = .101                 # 90 keV
        d_en = 3e-2
    else:
        raise ValueError('En is not in [30, 50, 70, 90] keV')

    bl.align_source(en, d_en)
    bl.align_crl(croc_crl_L, int(croc_crl_L), croc_crl_y_t, g_f, 0.)
    bl.align_crl_mask(100., 100.)
    bl.align_mono(en, r1, -6. * r1, r2, -6 * r2)

    metadata = check_repo(bl._metadata)
    with open(os.path.join(subdir, scan_name, 'md.csv'), 'w') as ff:
        ff.write('\n'.join('%s,%s' % (k, str(val))
                 for k, val in metadata.items()))

    for x_op in [24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40.]:
        bl.align_crl_mask(x_op, 100.)

        for plot in plts:
            if 'FM-XZ' in plot.title:
                plot.saveName = os.path.join(
                    subdir,
                    scan_name,
                    '%s-crl_mask_ox-%.03f.png' % (
                        plot.title, x_op)
                )
                plot.persistentName = plot.saveName.replace('.png', '.pickle')

                plot.xaxis.limits = [-.5, .5]
                plot.yaxis.limits = [-.3, .3]

        yield


def scan_lens_scale(bl: NSTU_SCW, plts: List):
    subdir = os.path.join(os.getenv('BASE_DIR', ''), 'datasets', 'nstu-scw-2')
    scan_name = 'scan_lens_scale'

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 30.e3
    if np.isclose(en, 30e3):
        r1, r2 = -2.04e3, -2.04e3  # 30 keV
        g_f = 1.076                # 30 keV
        d_en = 5e-3
    elif np.isclose(en, 50e3):
        r1, r2 = -1.22e3, -1.22e3  # 50 keV
        g_f = 0.390                # 50 keV
        d_en = 5e-3
    elif np.isclose(en, 70e3):
        r1, r2 = -.870e3, -.870e3  # 70 keV
        g_f = .191                 # 70 keV
        d_en = 1e-2
    elif np.isclose(en, 90e3):
        r1, r2 = -.675e3, -.675e3  # 90 keV
        g_f = .101                 # 90 keV
        d_en = 3e-2
    else:
        raise ValueError('En is not in [30, 50, 70, 90] keV')

    bl.align_source(en, d_en)
    bl.align_crl(croc_crl_L, int(croc_crl_L), g_f, g_f, 0.)
    bl.align_crl_mask(100., 100.)
    bl.align_mono(en, r1, -6. * r1, r2, -6 * r2)

    metadata = check_repo(bl._metadata)
    with open(os.path.join(subdir, scan_name, 'md.csv'), 'w') as ff:
        ff.write('\n'.join('%s,%s' % (k, str(val))
                 for k, val in metadata.items()))

    for scale in [.05, .1, .2, .3, .4]:  # [.5, .75, 1., 1.25, 1.5, 1.75, 2.]:
        bl.align_crl(
            croc_crl_L * scale,
            int(croc_crl_L * scale),
            croc_crl_y_t * np.sqrt(scale),
            g_f * np.sqrt(scale),
            0.
        )

        for plot in plts:
            if 'FM-XZ' in plot.title:
                plot.saveName = os.path.join(
                    subdir,
                    scan_name,
                    '%s-crl_l-%.03f-crl_d-%.03f.png' % (
                        plot.title, croc_crl_L * scale, croc_crl_y_t * np.sqrt(scale))
                )
                plot.persistentName = plot.saveName.replace('.png', '.pickle')

                plot.xaxis.limits = [-.5, .5]
                plot.yaxis.limits = [-.2, .2]

        yield


if __name__ == '__main__':
    beamline = NSTU_SCW()
    scan = onept 
    show = False
    repeats = 40

    if show:
        beamline.glow(
            scale=[1e3, 1e3, 1e3],
            centerAt=r'Si[111] Crystal 1',
            generator=scan,
            generatorArgs=[beamline, plots],
            startFrom=1
        )
    else:
        xrtrun.run_ray_tracing(
            beamLine=beamline,
            plots=plots,
            repeats=repeats,
            backend=r"raycing",
            generator=scan,
            generatorArgs=[beamline, plots]
        )
