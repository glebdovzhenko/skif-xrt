import os
import pickle
from typing import List
import numpy as np
import re
from matplotlib import pyplot as plt

import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from NSTU_SCW import NSTU_SCW
from components import PrismaticLens

from utils.xrtutils import (
    pickle_to_table,
)
from params.params_nstu_scw import (
    croc_geometry_68_percent,
    croc_geometry_87_percent,
    croc_geometry_95_percent,
    croc_geometry_BSU,
)

# matplotlib.use("agg")


# ############################## ONEPT SCAN ###################################
def onept_plots(*args, **kwargs):

    extra_kwargs = {
        r"fwhmFormatStr": "%.2e",
        "bins": 256,
        "ppb": 1,
    }

    x_kwds = {r"label": r"$x$", r"unit": r"mm", r"data": raycing.get_x, **extra_kwargs}
    y_kwds = {r"label": r"$y$", r"unit": r"mm", r"data": raycing.get_y, **extra_kwargs}
    z_kwds = {r"label": r"$z$", r"unit": r"mm", r"data": raycing.get_z, **extra_kwargs}
    xpr_kwds = {
        r"label": r"$x^{\prime}$",
        r"unit": r"",
        r"data": raycing.get_xprime,
        **extra_kwargs,
    }
    zpr_kwds = {
        r"label": r"$z^{\prime}$",
        r"unit": r"",
        r"data": raycing.get_zprime,
        **extra_kwargs,
    }

    result = []
    for beam, t1 in zip(
        (
            "BeamSourceLocal",
            "BeamAperture1Local",
            "BeamLensExitLocal",
            "BeamMonoC1Local",
            "BeamMonitor1Local",
            "BeamMonoC2Local",
            "BeamMonitor2Local",
        ),
        ("SRC", "FE", "CRL", "C1", "C1C2", "C2", "FM"),
    ):
        if t1 not in ("C1", "C2"):
            params = zip(
                ("XZ", "XXpr", "ZZpr"),
                (x_kwds, x_kwds, z_kwds),
                (z_kwds, xpr_kwds, zpr_kwds),
            )
        else:
            params = zip(("XY", "XXpr"), (x_kwds, x_kwds), (y_kwds, xpr_kwds))

        for t2, xkw, ykw in params:
            result.append(
                xrtplot.XYCPlot(
                    beam=beam,
                    title="-".join((t1, t2)),
                    xaxis=xrtplot.XYCAxis(**xkw),
                    yaxis=xrtplot.XYCAxis(**ykw),
                    aspect="auto",
                )
            )
    return result


def onept(bl: NSTU_SCW, plts: List):
    """
    30 keV, r1r2 = -2040, focal = 11500, offset=90
    50 keV, r1r2 = -1220, focal = 11425, offset=430
    70 keV, r1r2 = -870, focal = 12000, offset=955
    90 keV, r1r2 = -670, focal = 12300
    """

    bl.set_effective_filters()
    bl.align_90_keV()
    bl.SuperCWiggler.nrays = 100000

    for plot in plts:
        plot.xaxis.limits = None
        plot.yaxis.limits = None
        plot.caxis.limits = None

        # if "FM-XZ" in plot.title:
        #     plot.xaxis.limits = [-0.5, 0.5]
        #     plot.yaxis.limits = [-0.05 - 0.0644, 0.05 - 0.0644]

    yield


# ############################## y_g SCAN ###################################
def y_g_scan_plots(*args, **kwargs):

    extra_kwargs = {
        r"fwhmFormatStr": "%.2e",
        "bins": 256,
        "ppb": 1,
    }

    x_kwds = {r"label": r"$x$", r"unit": r"mm", r"data": raycing.get_x, **extra_kwargs}
    y_kwds = {r"label": r"$y$", r"unit": r"mm", r"data": raycing.get_y, **extra_kwargs}
    z_kwds = {r"label": r"$z$", r"unit": r"mm", r"data": raycing.get_z, **extra_kwargs}
    xpr_kwds = {
        r"label": r"$x^{\prime}$",
        r"unit": r"",
        r"data": raycing.get_xprime,
        **extra_kwargs,
    }
    zpr_kwds = {
        r"label": r"$z^{\prime}$",
        r"unit": r"",
        r"data": raycing.get_zprime,
        **extra_kwargs,
    }

    result = []
    for beam, t1 in (
        # ("BeamSourceLocal", "SRC"),
        # ("BeamAperture1Local", "FE"),
        ("BeamLensExitLocal", "CRL"),
        ("BeamMonoC1Local", "C1"),
        # ("BeamMonitor1Local", "C1C2"),
        ("BeamMonoC2Local", "C2"),
        ("BeamMonitor2Local", "FM"),
    ):
        if t1 not in ("C1", "C2"):
            params = zip(
                ("XZ", "XXpr", "ZZpr"),
                (x_kwds, x_kwds, z_kwds),
                (z_kwds, xpr_kwds, zpr_kwds),
            )
        else:
            params = zip(("XY", "XXpr"), (x_kwds, x_kwds), (y_kwds, xpr_kwds))

        for t2, xkw, ykw in params:
            result.append(
                xrtplot.XYCPlot(
                    beam=beam,
                    title="-".join((t1, t2)),
                    xaxis=xrtplot.XYCAxis(**xkw),
                    yaxis=xrtplot.XYCAxis(**ykw),
                    aspect="auto",
                )
            )
    return result


def y_g_scan(bl: NSTU_SCW, plts: List):
    """
    30 keV, r1r2 = -2040, focal = 11500
    50 keV, r1r2 = -1220, focal = 11425
    70 keV, r1r2 = -870, focal = 12000
    90 keV, r1r2 = -670, focal = 12300
    """
    en = 90.0e3
    bl.align_90_keV()
    # r1, r2 = -670, -670
    # d_en = 1.0
    # bl.align_source(en, d_en)
    bl.align_front_end()
    bl.ExitSlit.opening = [-100, 100, -100, 100]
    # bl.align_mono(en, r1, -6.0 * r1, r2, -6.0 * r2, c1_en_offset=90, c2_en_offset=-90)
    geom = croc_geometry_BSU["Be"]
    result = {"focal": [], "dz": []}
    for focal in np.linspace(9000, 14000, 15):
        y_g = PrismaticLens.calc_y_g(bl.LensMaterial, focal, en, geom["y_t"], geom["L"])
        bl.align_crl(L=geom["L"], N=geom["N"], d=geom["y_t"], g_f=y_g, g_l=0.0)
        bl.align_front_end(
            dzprime=2.0 * min(geom["y_t"], y_g) / bl.CrocLensStack[0].center[1]
        )
        for plot in plts:
            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None

            # if "FM-XZ" in plot.title:
            #     plot.xaxis.limits = [-0.5, 0.5]
            #     plot.yaxis.limits = [-0.5, 0.5]
        yield
        result["focal"].append(focal)
        print(f"Fdist = {focal}, y_g = {y_g}")
        for plot in plts:
            if "FM-XZ" in plot.title:
                print(f"Image {plot.dy}")
                result["dz"].append(plot.dy)
            # if "S-XZ" in plot.title:
            #     print(f"Source: {plot.dy}")

    fig = plt.figure()
    plt.plot(result["focal"], result["dz"])
    fig.show()


# ################################ r SCAN #####################################
def r_scan_plots(*args, **kwargs):

    extra_kwargs = {
        r"fwhmFormatStr": "%.2e",
        "bins": 256,
        "ppb": 1,
    }

    x_kwds = {r"label": r"$x$", r"unit": r"mm", r"data": raycing.get_x, **extra_kwargs}
    y_kwds = {r"label": r"$y$", r"unit": r"mm", r"data": raycing.get_y, **extra_kwargs}
    z_kwds = {r"label": r"$z$", r"unit": r"mm", r"data": raycing.get_z, **extra_kwargs}
    xpr_kwds = {
        r"label": r"$x^{\prime}$",
        r"unit": r"",
        r"data": raycing.get_xprime,
        **extra_kwargs,
    }
    zpr_kwds = {
        r"label": r"$z^{\prime}$",
        r"unit": r"",
        r"data": raycing.get_zprime,
        **extra_kwargs,
    }

    result = []
    for beam, t1 in (
        ("BeamSourceLocal", "SRC"),
        # ("BeamAperture1Local", "FE"),
        ("BeamLensExitLocal", "CRL"),
        # ("BeamMonoC1Local", "C1"),
        # ("BeamMonitor1Local", "C1C2"),
        # ("BeamMonoC2Local", "C2"),
        ("BeamMonitor2Local", "FM"),
    ):

        if t1 not in ("C1", "C2"):
            params = zip(
                ("XZ", "XXpr", "ZZpr"),
                (x_kwds, x_kwds, z_kwds),
                (z_kwds, xpr_kwds, zpr_kwds),
            )
        else:
            params = zip(("XY", "XXpr"), (x_kwds, x_kwds), (y_kwds, xpr_kwds))

        for t2, xkw, ykw in params:
            result.append(
                xrtplot.XYCPlot(
                    beam=beam,
                    title="-".join((t1, t2)),
                    xaxis=xrtplot.XYCAxis(**xkw),
                    yaxis=xrtplot.XYCAxis(**ykw),
                    aspect="auto",
                )
            )
    return result


def r_scan(bl: NSTU_SCW, plts: List):
    """
    30 keV r = -2040
    50 keV r = -1220
    70 keV r = -870
    90 keV r = -670
    """
    en = 90.0e3
    croc_focal = 11500
    d_en = 1.0
    geom = croc_geometry_87_percent["GC"]

    bl.align_source(en, d_en)
    bl.set_effective_filters()
    y_g = PrismaticLens.calc_y_g(
        bl.LensMaterial, croc_focal, en, geom["y_t"], geom["L"]
    )
    bl.align_crl(geom["L"], 100, geom["y_t"], y_g, 0.0)

    bl.align_front_end(
        dzprime=2.0 * min(geom["y_t"], y_g) / bl.CrocLensStack[0].center[1]
    )
    result = {"r": [], "dx": []}
    for r in np.linspace(-650, -700, 11):  # (-2e3) * np.linspace(0.8, 1.2, 11):
        bl.align_mono(en, r, -6.0 * r, r, -6.0 * r)
        for plot in plts:
            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None

            # if "FM-XZ" in plot.title:
            #     plot.yaxis.limits = [-0.1, 0.1]
        yield
        result["r"].append(r)
        print(f"r = {r}")
        for plot in plts:
            if "FM-XZ" in plot.title:
                print(f"Image {plot.dy}")
                print(f"Image {plot.dx}")
                result["dx"].append(plot.dx)
            # if "S-XZ" in plot.title:
            #     print(f"Source: {plot.dy}")

    fig = plt.figure()
    plt.plot(result["r"], result["dx"])
    fig.show()


# ########################## CROC GEOMETRY SCAN ###############################
def croc_scan_plots(*args, **kwargs):

    extra_kwargs = {
        r"fwhmFormatStr": "%.2e",
        "bins": 256,
        "ppb": 1,
    }

    x_kwds = {r"label": r"$x$", r"unit": r"mm", r"data": raycing.get_x, **extra_kwargs}
    y_kwds = {r"label": r"$y$", r"unit": r"mm", r"data": raycing.get_y, **extra_kwargs}
    z_kwds = {r"label": r"$z$", r"unit": r"mm", r"data": raycing.get_z, **extra_kwargs}
    xpr_kwds = {
        r"label": r"$x^{\prime}$",
        r"unit": r"",
        r"data": raycing.get_xprime,
        **extra_kwargs,
    }
    zpr_kwds = {
        r"label": r"$z^{\prime}$",
        r"unit": r"",
        r"data": raycing.get_zprime,
        **extra_kwargs,
    }

    result = []
    for beam, t1 in (
        ("BeamSourceLocal", "SRC"),
        ("BeamAperture1Local", "FE"),
        ("BeamLensExitLocal", "CRL"),
        # ("BeamMonoC1Local", "C1"),
        # ("BeamMonitor1Local", "C1C2"),
        # ("BeamMonoC2Local", "C2"),
        ("BeamMonitor2Local", "FM"),
    ):
        if t1 not in ("C1", "C2"):
            params = zip(
                ("XZ", "XXpr", "ZZpr"),
                (x_kwds, x_kwds, z_kwds),
                (z_kwds, xpr_kwds, zpr_kwds),
            )
        else:
            params = zip(("XY", "XXpr"), (x_kwds, x_kwds), (y_kwds, xpr_kwds))

        for t2, xkw, ykw in params:
            result.append(
                xrtplot.XYCPlot(
                    beam=beam,
                    title="-".join((t1, t2)),
                    xaxis=xrtplot.XYCAxis(**xkw),
                    yaxis=xrtplot.XYCAxis(**ykw),
                    aspect="auto",
                )
            )
    result.append(
        xrtplot.XYCPlot(
            beam="BeamMonitor2Local",
            title="FM-ZEn",
            xaxis=xrtplot.XYCAxis(**z_kwds),
            yaxis=xrtplot.XYCAxis(
                label=r"Energy",
                unit=r"eV",
                data=raycing.get_energy,
                **extra_kwargs,
            ),
            aspect="auto",
        )
    )
    return result


def croc_scan(bl: NSTU_SCW, plts: List):
    """
    30 keV croc_focal = 11500, r1r2 = -2040
    """
    en = 30.0e3
    r1, r2 = -2040, -2040  # 30 keV
    d_en = 150.0
    croc_focal = 11500
    bl.align_source(en, d_en)
    bl.align_mono(en, r1, -6.0 * r1, r2, -6.0 * r2, c1_en_offset=90, c2_en_offset=-90)

    bl.ExitSlit.center[2] -= 0.06
    bl.ExitSlit.opening = [-1, 1, -1, 1]

    for geom, dd in (
        (croc_geometry_68_percent["GC"], "p68"),
        (croc_geometry_87_percent["GC"], "p87"),
        (croc_geometry_95_percent["GC"], "p95"),
    ):
        y_g = PrismaticLens.calc_y_g(
            bl.LensMaterial, croc_focal, en, geom["y_t"], geom["L"]
        )
        bl.align_crl(geom["L"], 50, geom["y_t"], y_g, 0.0)
        bl.align_front_end(
            dzprime=min(geom["y_t"], y_g) / bl.CrocLensStack[0].center[1]
        )
        for plot in plts:
            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None
            # plot.saveName = (
            #     "/home/glebd/Dev/skif-xrt/datasets/nstu-scw/croc_len/"
            #     + dd
            #     + "/"
            #     + plot.title
            #     + ".png"
            # )

            # if "FM-XZ" in plot.title:
            # plot.xaxis.limits = [-1.0, 1.0]
            # plot.yaxis.limits = [-0.1 - 0.0532, 0.1 - 0.0532]
            # plot.yaxis.offset = -0.0532
        yield


# ########################## ABSORBED POWER SCAN ##############################

# for beam in [
#     "BeamFilterCLocal2a_{0:02d}".format(ii) for ii in range(diamond_filter_N)
# ] + ["BeamFilterSiCLocal2a_{0:02d}".format(ii) for ii in range(sic_filter_N)]:
#     t1 = beam.replace("BeamFilter", "").replace("Local2a_", "")
#     t2 = "XZ"
#     plots.append(
#         xrtplot.XYCPlot(
#             beam=beam,
#             title="-".join((t1, t2)),
#             xaxis=xrtplot.XYCAxis(
#                 limits=[-filter_size_x / 2, filter_size_x / 2], **x_kwds
#             ),
#             yaxis=xrtplot.XYCAxis(
#                 limits=[-filter_size_z / 2, filter_size_z / 2], **y_kwds
#             ),
#             fluxKind="power",
#             aspect="auto",
#         )
#     )


def absorbed_power_plots(*args, **kwargs):

    extra_kwargs = {
        r"fwhmFormatStr": "%.2e",
        "bins": 256,
        "ppb": 1,
    }

    x_kwds = {r"label": r"$x$", r"unit": r"mm", r"data": raycing.get_x, **extra_kwargs}
    y_kwds = {r"label": r"$y$", r"unit": r"mm", r"data": raycing.get_y, **extra_kwargs}
    z_kwds = {r"label": r"$z$", r"unit": r"mm", r"data": raycing.get_z, **extra_kwargs}

    result = []
    fmt1 = "BeamFilterCLocal2a_{0:02d}"
    fmt2 = "BeamFilterSiCLocal2a_{0:02d}"
    fmt3 = "BeamLensLocal2a_{0:02d}"
    # beams = (
    #     [fmt1.format(ii) for ii in range(5)]
    #     + [fmt2.format(ii) for ii in range(5)]
    #     + [fmt3.format(ii) for ii in range(5)]
    # )

    beams = [fmt3.format(ii) for ii in range(5)]

    for beam in beams:
        result.append(
            xrtplot.XYCPlot(
                beam=beam,
                title=beam,
                xaxis=xrtplot.XYCAxis(**x_kwds, limits=[-30, 30]),
                yaxis=xrtplot.XYCAxis(**y_kwds, limits=[-1, 1]),
                caxis=xrtplot.XYCAxis(label="Energy", limits=[100, 100000]),
                aspect="auto",
                fluxKind="power",
            )
        )
    return result


def absorbed_power(bl: NSTU_SCW, plts: List):
    bl.set_filter_stacks()
    bl.align_90_keV()
    # bl.align_front_end()
    bl.SuperCWiggler.eMin = 10.0
    bl.SuperCWiggler.eMax = 100000.0
    bl.SuperCWiggler.eN = 10001
    print(bl.FilterEffectiveC, len(bl.FilterStackC))
    print(bl.FilterEffectiveSiC, len(bl.FilterStackSiC))
    yield

    # for plot in plts:
    #     if plot.persistentName is not None:
    #         with open(plot.persistentName, "rb") as f:
    #             f = pickle.load(f)
    #             np.savetxt(
    #                 plot.persistentName.replace(".pickle", ".txt"),
    #                 pickle_to_table(f),
    #                 delimiter=" ",
    #                 header="""\"x (mm)\"	\"y (mm)\"	\"Filtered Power (W/mm<sup>2</sup>)\"""",
    #             )


# ################################## RUN ######################################
if __name__ == "__main__":
    beamline = NSTU_SCW()
    scan = absorbed_power
    plot_gen = absorbed_power_plots
    show = False
    repeats = 2

    if show:
        beamline.glow(
            scale=[1e3, 1e3, 1e3],
            generator=scan,
            generatorArgs=[beamline, []],
            startFrom=1,
        )
    else:
        plots = plot_gen(beamline)
        xrtrun.run_ray_tracing(
            beamLine=beamline,
            plots=plots,
            repeats=repeats,
            backend=r"raycing",
            generator=scan,
            generatorArgs=[beamline, plots],
        )
