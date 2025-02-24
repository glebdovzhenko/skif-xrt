from typing import List
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib

matplotlib.use("agg")

import xrt.runner as xrtrun
import xrt.plotter as xrtplot
import xrt.backends.raycing as raycing

from utils.xrtutils import get_line_kb, get_minmax

from SKIF_1_5 import SKIF15


plots = []
x_kwds = {r"label": r"$x$", r"unit": r"mm", r"data": raycing.get_x}
z_kwds = {r"label": r"$z$", r"unit": r"mm", r"data": raycing.get_z}
xpr_kwds = {r"label": r"$x^{\prime}$", r"unit": r"", r"data": raycing.get_xprime}
zpr_kwds = {r"label": r"$z^{\prime}$", r"unit": r"", r"data": raycing.get_zprime}

for beam, t1 in zip(
    (
        "BeamAperture1Local",
        "BeamMonoC1Local",
        "BeamMonitor1Local",
        "BeamMonoC2Local",
        "BeamFocusingMonitorLocal",
        "BeamMonitor2Local",
        "BeamAperture2Local",
    ),
    ("FE", "C1", "C1C2", "C2", "FM", "EM", "ES"),
):
    for t2, xkw, ykw in zip(
        ("XZ", "XXpr", "ZZpr"), (x_kwds, x_kwds, z_kwds), (z_kwds, xpr_kwds, zpr_kwds)
    ):
        plots.append(
            xrtplot.XYCPlot(
                beam=beam,
                title="-".join((t1, t2)),
                xaxis=xrtplot.XYCAxis(**xkw),
                yaxis=xrtplot.XYCAxis(**ykw),
                aspect="auto",
            )
        )

# for beam in sum([[b_name % (ii + 1) for ii in range(11)] for b_name in
#                  ('BeamFilter%dLocal1', 'BeamFilter%dLocal2', 'BeamFilter%dLocal2a')], []):

#     t1 = beam.replace('Local1', '').replace('Local2a', '').replace('Local2', '').replace('BeamFilter', 'F')

#     if beam[-6:] == 'Local1':
#         plots.append(xrtplot.XYCPlot(beam=beam, title=(t1+'I-XZ'), aspect='auto',
#                             xaxis=xrtplot.XYCAxis(label='$x$', unit='mm', data=raycing.get_x),
#                             yaxis=xrtplot.XYCAxis(label='$y$', unit='mm', data=raycing.get_y)))
#         plots.append(xrtplot.XYCPlot(beam=beam, title=(t1+'P-XZ'), aspect='auto', fluxKind='power',
#                             xaxis=xrtplot.XYCAxis(label='$x$', unit='mm', data=raycing.get_x),
#                             yaxis=xrtplot.XYCAxis(label='$y$', unit='mm', data=raycing.get_y)))
#     elif beam[-6:] == 'Local2':
#         plots.append(xrtplot.XYCPlot(beam=beam, title=(t1+'IT-XZ'), aspect='auto',
#                             xaxis=xrtplot.XYCAxis(label='$x$', unit='mm', data=raycing.get_x),
#                             yaxis=xrtplot.XYCAxis(label='$y$', unit='mm', data=raycing.get_y)))
#     elif beam[-7:] == 'Local2a':
#         plots.append(xrtplot.XYCPlot(beam=beam, title=(t1+'PA-XZ'), aspect='auto', fluxKind='power',
#                             xaxis=xrtplot.XYCAxis(label='$x$', unit='mm', data=raycing.get_x),
#                             yaxis=xrtplot.XYCAxis(label='$y$', unit='mm', data=raycing.get_y)))
#     else:
#         pass
# else:
#     plots.append(xrtplot.XYCPlot(beam='BeamMonoC1Local2a', title=('C1PA-XZ'), aspect='auto', fluxKind='power',
#                  xaxis=xrtplot.XYCAxis(label='$x$', unit='mm', data=raycing.get_x),
#                  yaxis=xrtplot.XYCAxis(label='$y$', unit='mm', data=raycing.get_y)))


def onept(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "test"

    r = 3.5e3
    en = 30.0e3
    bl.MonochromatorCr1.Rx = -300.0
    bl.MonochromatorCr2.Rx = -300

    bl.align_energy(en, bl.get_de_over_e(r, en))
    bl.set_plot_limits(plts)

    yield


def get_focus(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "get_focus_m_test_inf"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))
    # else:
    #     for f_name in os.listdir(os.path.join(subdir, scan_name)):
    #         os.remove(os.path.join(subdir, scan_name, f_name))

    en = 30.0e3
    # for r in [np.inf, -np.inf, .5, -.5, 1., -1., 1.5, -1.5, 2., -2., 2.5, -2.5, 3., -3., 3.5, -3.5, 4., -4.,
    #           4.5, -4.5, 5., -5., 6., -6., 7., -7., 8., -8., 9., -9., 10., -10., 11., -11., 12., -12.,
    #           13., -13., 14., -14., 15., -15., 16., -16., 17., -17., 18., -18., 19., -19., 20., -20.,
    #           21., -21., 25., -25., 29., -29., -1.6, -1.61, -1.62, -1.63, -1.64, -1.65, -1.66, -1.67,
    #           -1.68, -1.69, -1.7, -1.71, -1.72, -1.73, -1.74, -1.75, -1.76, -1.77, -1.78, -1.79, -1.8,
    #           -1.731, -1.732, -1.733, -1.734, -1.735, -1.736, -1.737, -1.738, -1.739,
    #           -1.741, -1.742, -1.743, -1.744, -1.745, -1.746, -1.747, -1.748, -1.749]:  # sagittal
    # for r in [np.inf, -np.inf, -140., -130., -120., -110., -100., -90., -80., -70., -60., -50., -40.,
    #           -30., -20., -10., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110.,
    #           120., 130., 140., 150., 160., 170., 180., 190., 200., 210., 220., 230., 240.,
    #           250., 260., 270., 280., 290., 300.]: # meridional

    bl.MonochromatorCr2.Ry = np.inf
    for r in [-2.1, -2.3, -2.5, -2.7, -2.45, -2.55, -2.65, -2.45]:
        # bl.MonochromatorCr1.Ry = 1e3 * r
        bl.MonochromatorCr1.Ry = 1e3 * r
        # bl.MonochromatorCr2.Ry = 1e3 * r
        # bl.MonochromatorCr2.Ry = -6e3 * r
        bl.align_energy(en, 1e-1)

        for plot in plts:
            el, crd = plot.title.split("-")
            if (el not in ("EM", "FM", "C1C2")) or (crd not in ("XZ", "XXpr", "ZZpr")):
                continue

            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None
            plot.saveName = os.path.join(
                subdir,
                scan_name,
                plot.title + "-%sm" % bl.MonochromatorCr1.pretty_R() + ".png",
            )
            plot.persistentName = plot.saveName.replace(".png", ".pickle")

        yield

        bl.align_energy(en, 1e-10)
        ax_mul = 1.1
        for plot in plts:
            el, crd = plot.title.split("-")
            if plot.saveName:
                data = pickle.load(open(plot.persistentName, "rb"))

                x1, x2 = get_minmax(data, axis="x")
                y1, y2 = get_minmax(data, axis="y")
                e1, e2 = get_minmax(data, axis="e")

                x1, x2 = x2 - ax_mul * np.abs(x1 - x2), x1 + ax_mul * np.abs(x1 - x2)
                y1, y2 = y2 - ax_mul * np.abs(y1 - y2), y1 + ax_mul * np.abs(y1 - y2)
                e1, e2 = e2 - ax_mul * np.abs(e1 - e2), e1 + ax_mul * np.abs(e1 - e2)

                plot.xaxis.limits = [x1, x2]
                plot.yaxis.limits = [y1, y2]
                plot.caxis.limits = [e1, e2]

                if bl.SuperCWiggler.eMin > e1:
                    bl.SuperCWiggler.eMin = e1
                if bl.SuperCWiggler.eMax < e2:
                    bl.SuperCWiggler.eMax = e2

                # os.remove(plot.persistentName)
                # os.remove(plot.saveName)
                plot.persistentName = plot.persistentName.replace(
                    scan_name, scan_name + "_"
                )
                plot.saveName = plot.saveName.replace(scan_name, scan_name + "_")

        yield


def e_scan(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "e_scan"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    with open(os.path.join(subdir, scan_name + ".txt"), "w") as f:
        tmp = bl.dumps()
        tmp += "METADATA\n"
        tmp += "SCAN: E\n"
        tmp += "FILES: NAME-E-R1-R2\n"
        f.write(tmp)

    for en in np.arange(30.0, 150.0, 5.0) * 1e3:
        bl.MonochromatorCr1.R = 57.5e3
        bl.MonochromatorCr2.R = 57.5e3

        bl.align_energy(en, bl.get_de_over_e(57.5e3, en))
        bl.set_plot_limits(plts)

        for plot in plts:
            plot.saveName = "%s-%dkeV-%sm-%sm.png" % (
                os.path.join(subdir, scan_name, plot.title),
                int(en * 1e-3),
                bl.MonochromatorCr1.pretty_R(),
                bl.MonochromatorCr2.pretty_R(),
            )
            plot.persistentName = plot.saveName.replace(".png", ".pickle")
        yield


def chi_scan(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "chi_scan"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    with open(os.path.join(subdir, scan_name + ".txt"), "w") as f:
        tmp = bl.dumps()
        tmp += "METADATA\n"
        tmp += "SCAN: CHI\n"
        tmp += "FILES: NAME-E-CHI-R1-R2\n"
        f.write(tmp)

    bl.MonochromatorCr1.R = 30.0e3
    bl.MonochromatorCr2.R = 30.0e3

    for en in [30.0e3, 90.0e3, 150.0e3]:

        for chi in [5.0, 35.0, 0.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
            bl.MonochromatorCr1.set_alpha(np.radians(chi))
            bl.MonochromatorCr2.set_alpha(-np.radians(chi))

            bl.align_energy(en, bl.get_de_over_e(20.0e3, en))
            bl.set_plot_limits(plts)

            for plot in plts:
                plot.saveName = "%s-%dkeV-%ddeg-%sm-%sm.png" % (
                    os.path.join(subdir, scan_name, plot.title),
                    int(en * 1e-3),
                    int(chi),
                    bl.MonochromatorCr1.pretty_R(),
                    bl.MonochromatorCr2.pretty_R(),
                )
                plot.persistentName = plot.saveName.replace(".png", ".pickle")
            yield

        for chi in [40.0, 45.0, 50.0, 55.0, 60.0]:
            bl.MonochromatorCr1.set_alpha(np.radians(chi))
            bl.MonochromatorCr2.set_alpha(-np.radians(chi))

            bl.align_energy(en, bl.get_de_over_e(12.0e3, en))
            bl.set_plot_limits(plts)

            for plot in plts:
                plot.saveName = "%s-%dkeV-%ddeg-%sm-%sm.png" % (
                    os.path.join(subdir, scan_name, plot.title),
                    int(en * 1e-3),
                    int(chi),
                    bl.MonochromatorCr1.pretty_R(),
                    bl.MonochromatorCr2.pretty_R(),
                )
                plot.persistentName = plot.saveName.replace(".png", ".pickle")
            yield


def t_scan(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "t_scan"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    with open(os.path.join(subdir, scan_name + ".txt"), "w") as f:
        tmp = bl.dumps()
        tmp += "METADATA\n"
        tmp += "SCAN: T\n"
        tmp += "FILES: NAME-E-T-R1-R2\n"
        f.write(tmp)

    bl.MonochromatorCr1.R = 30.0e3
    bl.MonochromatorCr2.R = 30.0e3
    bl.MonochromatorCr1.set_alpha(np.radians(35.3))
    bl.MonochromatorCr2.set_alpha(-np.radians(35.3))

    for en in [30.0e3, 90.0e3, 150.0e3]:
        for t in [
            2.5,
            3.0,
            3.5,
            4.0,
        ]:  # [1.25, 1.75, 2.25, 2.5, 0.5, 2.5, 1.0, 1.5, 2.0]:
            bl.MonochromatorCr1.material[0].t = t
            bl.MonochromatorCr2.material[0].t = t

            bl.align_energy(en, bl.get_de_over_e(20.0e3, en))
            bl.set_plot_limits(plts)

            for plot in plts:
                plot.saveName = "%s-%dkeV-%.01fmm-%sm-%sm.png" % (
                    os.path.join(subdir, scan_name, plot.title),
                    int(en * 1e-3),
                    t,
                    bl.MonochromatorCr1.pretty_R(),
                    bl.MonochromatorCr2.pretty_R(),
                )
                plot.persistentName = plot.saveName.replace(".png", ".pickle")
            yield


def r_scan(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "r_scan"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    with open(os.path.join(subdir, scan_name + ".txt"), "w") as f:
        tmp = bl.dumps()
        tmp += "METADATA\n"
        tmp += "SCAN: R\n"
        tmp += "FILES: NAME-E-T-R1-R2\n"
        f.write(tmp)

    bl.MonochromatorCr1.set_alpha(np.radians(35.3))
    bl.MonochromatorCr2.set_alpha(-np.radians(35.3))
    bl.MonochromatorCr1.material[0].t = 1.8
    bl.MonochromatorCr2.material[0].t = 2.2

    for en, ml in zip([30.0e3, 90.0e3, 150.0e3], [3.0, 4.0, 4.0]):
        for r in [60.0e3, 70.0e3, 80.0e3, 90.0e3, 100.0e3]:
            bl.MonochromatorCr1.R = r
            bl.MonochromatorCr2.R = r

            bl.align_energy(en, ml * bl.get_de_over_e(r, en))
            bl.set_plot_limits(plts)

            for plot in plts:
                plot.saveName = "%s-%dkeV-%sm-%sm.png" % (
                    os.path.join(subdir, scan_name, plot.title),
                    int(en * 1e-3),
                    bl.MonochromatorCr1.pretty_R(),
                    bl.MonochromatorCr2.pretty_R(),
                )
                plot.persistentName = plot.saveName.replace(".png", ".pickle")
            yield


def r_map(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "r_map"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    with open(os.path.join(subdir, scan_name + ".txt"), "w") as f:
        tmp = bl.dumps()
        tmp += "METADATA\n"
        tmp += "SCAN: R\n"
        tmp += "FILES: NAME-E-T-R1-R2\n"
        f.write(tmp)

    bl.MonochromatorCr1.set_alpha(np.radians(35.3))
    bl.MonochromatorCr2.set_alpha(-np.radians(35.3))
    bl.MonochromatorCr1.material[0].t = 1.8
    bl.MonochromatorCr2.material[0].t = 2.2

    for en, ml in zip([30.0e3, 90.0e3, 150.0e3], [1.0, 3.0, 3.0]):
        for r1 in [5.0e3, 10.0e3, 20.0e3, 30.0e3, 40.0e3, 50.0e3]:
            for r2 in np.linspace(r1 - 2.0e3, r1 + 2.0e3, 11):

                bl.MonochromatorCr1.R = r1
                bl.MonochromatorCr2.R = r2

                bl.align_energy(en, 2.0 * ml * bl.get_de_over_e(r1, en))
                bl.set_plot_limits(plts)

                for plot in plts:
                    plot.saveName = "%s-%dkeV-%sm-%sm.png" % (
                        os.path.join(subdir, scan_name, plot.title),
                        int(en * 1e-3),
                        bl.MonochromatorCr1.pretty_R(),
                        bl.MonochromatorCr2.pretty_R(),
                    )
                    plot.persistentName = plot.saveName.replace(".png", ".pickle")
                yield


def tth_offset_scan(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "tth_offset_scan2"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    with open(os.path.join(subdir, scan_name + ".txt"), "w") as f:
        tmp = bl.dumps()
        tmp += "METADATA\n"
        tmp += "SCAN: T\n"
        tmp += "FILES: NAME-E-TTH-R1-R2\n"
        f.write(tmp)

    bl.MonochromatorCr1.R = 30.0e3
    bl.MonochromatorCr2.R = 30.0e3
    bl.MonochromatorCr1.set_alpha(np.radians(35.3))
    bl.MonochromatorCr2.set_alpha(-np.radians(35.3))

    for en, ml in zip([30.0e3, 90.0e3, 150.0e3], [2.0, 2.0, 2.0]):
        for r in [np.inf, 20.0e3]:
            bl.MonochromatorCr1.R = r
            bl.MonochromatorCr2.R = r

            bl.align_energy(en, ml * bl.get_de_over_e(r, en))
            bl.set_plot_limits(plts)

            tths = np.concatenate(
                (
                    np.linspace(-70.0, 70.0, 40) * 1e-6,
                    np.linspace(-10.0, 10.0, 20) * 1e-6,
                )
            )
            tths = np.linspace(-2.5, 2.5, 40) * 1e-6

            for tth in tths:
                bl.MonochromatorCr1.extraPitch = tth

                for plot in plts:
                    plot.saveName = "%s-%dkeV-%.03farcsec-%sm-%sm.png" % (
                        os.path.join(subdir, scan_name, plot.title),
                        int(en * 1e-3),
                        np.degrees(tth) * 3600.0,
                        bl.MonochromatorCr1.pretty_R(),
                        bl.MonochromatorCr2.pretty_R(),
                    )
                    plot.persistentName = plot.saveName.replace(".png", ".pickle")
                yield


def roll_offset_scan(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "roll_offset_scan"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    with open(os.path.join(subdir, scan_name + ".txt"), "w") as f:
        tmp = bl.dumps()
        tmp += "METADATA\n"
        tmp += "SCAN: T\n"
        tmp += "FILES: NAME-E-TTH-R1-R2\n"
        f.write(tmp)

    bl.MonochromatorCr1.R = 30.0e3
    bl.MonochromatorCr2.R = 30.0e3
    bl.MonochromatorCr1.set_alpha(np.radians(35.3))
    bl.MonochromatorCr2.set_alpha(-np.radians(35.3))

    bl.MonochromatorCr1.rotationSequence = "RyRxRz"

    for en, ml in zip([30.0e3, 90.0e3, 150.0e3], [2.0, 2.0, 2.0]):
        for r in [np.inf, 20.0e3]:
            bl.MonochromatorCr1.R = r
            bl.MonochromatorCr2.R = r

            bl.align_energy(en, ml * bl.get_de_over_e(r, en))
            bl.set_plot_limits(plts)

            # tths = np.concatenate((
            #     np.linspace(-70., 70., 40) * 1e-6,
            #     np.linspace(-10., 10., 20) * 1e-6))
            tths = np.concatenate(
                (np.linspace(-8.5, 8.5, 40) * 1e-2, np.linspace(-8.5, 8.5, 40) * 1e-3)
            )

            for tth in tths:
                bl.MonochromatorCr1.roll = tth

                for plot in plts:
                    plot.saveName = "%s-%dkeV-%.03farcsec-%sm-%sm.png" % (
                        os.path.join(subdir, scan_name, plot.title),
                        int(en * 1e-3),
                        np.degrees(tth) * 3600.0,
                        bl.MonochromatorCr1.pretty_R(),
                        bl.MonochromatorCr2.pretty_R(),
                    )
                    plot.persistentName = plot.saveName.replace(".png", ".pickle")
                yield


def yaw_offset_scan(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "yaw_offset_scan"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    with open(os.path.join(subdir, scan_name + ".txt"), "w") as f:
        tmp = bl.dumps()
        tmp += "METADATA\n"
        tmp += "SCAN: T\n"
        tmp += "FILES: NAME-E-TTH-R1-R2\n"
        f.write(tmp)

    bl.MonochromatorCr1.R = 30.0e3
    bl.MonochromatorCr2.R = 30.0e3
    bl.MonochromatorCr1.set_alpha(np.radians(35.3))
    bl.MonochromatorCr2.set_alpha(-np.radians(35.3))

    bl.MonochromatorCr1.rotationSequence = "RzRxRy"

    for en, ml in zip([30.0e3, 90.0e3, 150.0e3], [2.0, 2.0, 2.0]):
        for r in [np.inf, 20.0e3]:
            bl.MonochromatorCr1.R = r
            bl.MonochromatorCr2.R = r

            bl.align_energy(en, ml * bl.get_de_over_e(r, en))
            bl.set_plot_limits(plts)

            # tths = np.concatenate((
            #     np.linspace(-70., 70., 40) * 1e-6,
            #     np.linspace(-10., 10., 20) * 1e-6))
            tths = np.concatenate(
                (np.linspace(-8.5, 8.5, 40) * 1e-2, np.linspace(-8.5, 8.5, 40) * 1e-3)
            )

            for tth in tths:
                bl.MonochromatorCr1.yaw = tth

                for plot in plts:
                    plot.saveName = "%s-%dkeV-%.03farcsec-%sm-%sm.png" % (
                        os.path.join(subdir, scan_name, plot.title),
                        int(en * 1e-3),
                        np.degrees(tth) * 3600.0,
                        bl.MonochromatorCr1.pretty_R(),
                        bl.MonochromatorCr2.pretty_R(),
                    )
                    plot.persistentName = plot.saveName.replace(".png", ".pickle")
                yield


def y_offset_scan(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "y_offset_scan"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    with open(os.path.join(subdir, scan_name + ".txt"), "w") as f:
        tmp = bl.dumps()
        tmp += "METADATA\n"
        tmp += "SCAN: T\n"
        tmp += "FILES: NAME-E-TTH-R1-R2\n"
        f.write(tmp)

    bl.MonochromatorCr1.R = 30.0e3
    bl.MonochromatorCr2.R = 30.0e3
    bl.MonochromatorCr1.set_alpha(np.radians(35.3))
    bl.MonochromatorCr2.set_alpha(-np.radians(35.3))

    for en, ml in zip([30.0e3, 90.0e3, 150.0e3], [2.0, 2.0, 2.0]):
        for r in [np.inf, 20.0e3]:
            bl.MonochromatorCr1.R = r
            bl.MonochromatorCr2.R = r

            bl.align_energy(en, ml * bl.get_de_over_e(r, en))
            bl.set_plot_limits(plts)

            # tths = np.concatenate((
            #     np.linspace(-70., 70., 40) * 1e-6,
            #     np.linspace(-10., 10., 20) * 1e-6))
            # tths = np.linspace(-2.5, 2.5, 40) * 1e-6

            dys = np.linspace(-10.0, 10.0, 50)

            for dy in dys:
                bl.align_energy(en, ml * bl.get_de_over_e(r, en))
                bl.MonochromatorCr2.center[1] += dy

                for plot in plts:
                    plot.saveName = "%s-%dkeV-%.03fdmcm-%sm-%sm.png" % (
                        os.path.join(subdir, scan_name, plot.title),
                        int(en * 1e-3),
                        dy * 1e3,
                        bl.MonochromatorCr1.pretty_R(),
                        bl.MonochromatorCr2.pretty_R(),
                    )
                    plot.persistentName = plot.saveName.replace(".png", ".pickle")
                yield


def z_offset_scan(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "test"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    with open(os.path.join(subdir, scan_name + ".txt"), "w") as f:
        tmp = bl.dumps()
        tmp += "METADATA\n"
        tmp += "SCAN: T\n"
        tmp += "FILES: NAME-E-TTH-R1-R2\n"
        f.write(tmp)

    bl.MonochromatorCr1.R = 30.0e3
    bl.MonochromatorCr2.R = 30.0e3
    bl.MonochromatorCr1.set_alpha(np.radians(35.3))
    bl.MonochromatorCr2.set_alpha(-np.radians(35.3))

    for en, ml in zip([30.0e3, 90.0e3, 150.0e3], [2.0, 2.0, 2.0]):
        for r in [np.inf, 20.0e3]:
            bl.MonochromatorCr1.R = r
            bl.MonochromatorCr2.R = r

            bl.align_energy(en, ml * bl.get_de_over_e(r, en))
            bl.set_plot_limits(plts)

            # tths = np.concatenate((
            #     np.linspace(-70., 70., 40) * 1e-6,
            #     np.linspace(-10., 10., 20) * 1e-6))
            # tths = np.linspace(-2.5, 2.5, 40) * 1e-6

            dzs = np.linspace(-2.0, 2.0, 50)

            for dz in dzs:
                bl.align_energy(en, ml * bl.get_de_over_e(r, en))
                bl.MonochromatorCr2.center[2] += dz

                for plot in plts:
                    plot.saveName = "%s-%dkeV-%.03fdmcm-%sm-%sm.png" % (
                        os.path.join(subdir, scan_name, plot.title),
                        int(en * 1e-3),
                        dz * 1e3,
                        bl.MonochromatorCr1.pretty_R(),
                        bl.MonochromatorCr2.pretty_R(),
                    )
                    plot.persistentName = plot.saveName.replace(".png", ".pickle")
                yield


def calc_abs(plts: List, bl: SKIF15):
    subdir = r"/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/datasets/skif15"
    scan_name = "absorption"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    r = np.inf
    en = 30.0e3
    bl.MonochromatorCr1.R = r
    bl.MonochromatorCr2.R = r

    bl.align_energy(en, bl.get_de_over_e(r, en))
    bl.SuperCWiggler.eMin, bl.SuperCWiggler.eMax = 10.0, 500000.0
    bl.SuperCWiggler.nrays = 10000
    # bl.set_plot_limits(plts)

    for plt in plts:
        plt.caxis.offset = 0.0
        plt.caxis.limits = [0.0, 100000.0]

    for plot in plts:
        plot.saveName = "%s.png" % os.path.join(subdir, scan_name, plot.title)
        plot.persistentName = plot.saveName.replace(".png", ".pickle")

    yield


def bump_size_scan(plts: List, bl: SKIF15):
    subdir = os.path.join(os.getenv("BASE_DIR"), "datasets", "skif15")
    scan_name = "bump_size_flat"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 30.0e3
    bl.MonochromatorCr1.R = np.inf  # 20e3  # np.inf
    bl.MonochromatorCr2.R = np.inf  # 20e3  # np.inf

    bl.align_energy(en, 700.0 / 30.0e3)

    for amp in [0.0]:  # np.logspace(np.log10(0.0007), np.log10(0.003), 25):
        bl.MonochromatorCr1.bump_pars = {
            "Cx": 0.0,
            "Cy": 0.0,
            "Sx": 23.0,
            "Sy": 5.0,
            "Axy": amp,
        }
        for plot in plts:
            plot.saveName = "%s-%dkeV-%f.png" % (
                os.path.join(subdir, scan_name, plot.title),
                int(en * 1e-3),
                amp,
            )
            plot.persistentName = plot.saveName.replace(".png", ".pickle")

            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None
        yield


def bump_r2_scan(plts: List, bl: SKIF15):
    subdir = os.path.join(os.getenv("BASE_DIR"), "datasets", "skif15")
    scan_name = "bump_r2_scan"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 30.0e3
    bl.MonochromatorCr1.R = 20e3  # np.inf
    bl.MonochromatorCr2.R = 20e3  # np.inf

    bl.align_energy(en, 700.0 / 30.0e3)
    bl.align_energy_c2(en - 44.33)

    bl.MonochromatorCr1.bump_pars = {
        "Cx": 0.0,
        "Cy": 0.0,
        "Sx": 23.0,
        "Sy": 5.0,
        "Axy": 3e-3,
    }

    for dr in [6.5e3]:
        bl.MonochromatorCr2.R = 20e3 + dr

        for plot in plts:
            plot.saveName = "%s-%dkeV-%s.png" % (
                os.path.join(subdir, scan_name, plot.title),
                int(en * 1e-3),
                bl.MonochromatorCr2.pretty_R(),
            )
            plot.persistentName = plot.saveName.replace(".png", ".pickle")

            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None
        yield


def bump_tth_scan(plts: List, bl: SKIF15):
    subdir = os.path.join(os.getenv("BASE_DIR"), "datasets", "skif15")
    scan_name = "bump_tth_scan2"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 30.0e3
    bl.MonochromatorCr1.R = 20e3  # np.inf
    bl.MonochromatorCr2.R = 5e3  # np.inf

    bl.align_energy(en, 700.0 / 30.0e3)
    bl.MonochromatorCr1.bump_pars = {
        "Cx": 0.0,
        "Cy": 0.0,
        "Sx": 23.0,
        "Sy": 5.0,
        "Axy": 3e-3,
    }

    for d_en in np.linspace(-200, 0, 20):
        bl.align_energy_c2(en + d_en)

        for plot in plts:
            plot.saveName = "%s-%dkeV-%f.png" % (
                os.path.join(subdir, scan_name, plot.title),
                int(en * 1e-3),
                d_en,
            )
            plot.persistentName = plot.saveName.replace(".png", ".pickle")

            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None
        yield


def bumb_tth_r2_map(plts: List, bl: SKIF15):
    subdir = os.path.join(os.getenv("BASE_DIR"), "datasets", "skif15")
    scan_name = "bump_tth_scan2"

    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    en = 30.0e3
    bl.MonochromatorCr1.R = 20e3  # np.inf
    bl.MonochromatorCr2.R = 5e3  # np.inf

    bl.align_energy(en, 700.0 / 30.0e3)
    bl.MonochromatorCr1.bump_pars = {
        "Cx": 0.0,
        "Cy": 0.0,
        "Sx": 23.0,
        "Sy": 5.0,
        "Axy": 3e-3,
    }


if __name__ == "__main__":
    beamline = SKIF15()
    scan = bump_r2_scan
    show = True
    repeats = 10

    if show:
        beamline.glow(
            scale=[1e3, 1e3, 1e3],
            centerAt=r"Si[111] Crystal 1",
            generator=scan,
            generatorArgs=[plots, beamline],
            startFrom=1,
        )
    else:
        xrtrun.run_ray_tracing(
            beamLine=beamline,
            plots=plots,
            repeats=repeats,
            backend=r"raycing",
            generator=scan,
            generatorArgs=[plots, beamline],
        )
