from typing import List
import numpy as np
from matplotlib import pyplot as plt

import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun
import xrt.backends.raycing.materials_elemental as rm

from components import PrismaticLens
from veips_branch import SURFACE_VEIPS


# ############################## DEFINITIONS ##################################
extra_kwargs = {
    r"fwhmFormatStr": "%.2e",
    "bins": 512,
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
e_kwds = {
    r"label": r"e",
    r"unit": r"keV",
    r"data": raycing.get_energy,
    **extra_kwargs,
}


# ############################## ONEPT SCAN ###################################
def onept_plots(*args, **kwargs):
    """"""
    result = []
    for beam, t1, y_k in zip(
        (
            "BeamSourceLocal",
            "BeamCRLEntranceLocal",
            "BeamCRLExitLocal",
            "BeamMonoLocal",
            "BeamCRLFocusLocal",
        ),
        (
            "SRC",
            "CRL Entrance",
            "CRL Exit",
            "Mono Surface",
            "BL Focus",
        ),
        (
            z_kwds,
            z_kwds,
            z_kwds,
            y_kwds,
            z_kwds,
        ),
    ):
        result.append(
            xrtplot.XYCPlot(
                beam=beam,
                title=t1,
                xaxis=xrtplot.XYCAxis(**x_kwds),
                yaxis=xrtplot.XYCAxis(**y_k),
                caxis=xrtplot.XYCAxis(**e_kwds),
                aspect="auto",
            )
        )

    return result


def onept(bl: SURFACE_VEIPS, plts: List):
    """"""
    bl.bm.eMin = bl.alignE * (1.0 - 1.5e-3)
    bl.bm.eMax = bl.alignE * (1.0 + 1.5e-3)
    bl.CrlFocusApt.opening = [-0.5, 0.5, -0.5, 0.5]
    bl.CrlEntranceApt.opening = [-70, 70, -1.75, 1.75]

    for plot in plts:
        if plot.title == "CRL Focus":
            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None
        else:
            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None
    yield


# ############################ ABSORPTION SCAN ################################
def absorption_plots(*args, **kwargs):
    """"""
    result = []
    for beam, t1, y_k in zip(
        (
            "BeamSourceGlobal",
            "BeamFilter1Local2",
            "BeamFilter2Local2",
        ),
        (
            "SRC",
            "Filter 1",
            "Filter 2",
        ),
        (
            z_kwds,
            y_kwds,
            y_kwds,
        ),
    ):
        result.append(
            xrtplot.XYCPlot(
                beam=beam,
                title=t1,
                xaxis=xrtplot.XYCAxis(**x_kwds),
                yaxis=xrtplot.XYCAxis(**y_k),
                caxis=xrtplot.XYCAxis(**e_kwds),
                aspect="auto",
                fluxKind="power",
            )
        )

    for ii in range(10):
        result.append(
            xrtplot.XYCPlot(
                beam="BeamLensLocal2_{0:02d}".format(ii),
                title="BeamLensLocal2_{0:02d}".format(ii),
                xaxis=xrtplot.XYCAxis(**x_kwds),
                yaxis=xrtplot.XYCAxis(**y_kwds),
                caxis=xrtplot.XYCAxis(**e_kwds),
                aspect="auto",
                fluxKind="power",
            )
        )

    result.append(
        xrtplot.XYCPlot(
            beam="BeamMonoLocal2",
            title="BeamMonoLocal2",
            xaxis=xrtplot.XYCAxis(**x_kwds),
            yaxis=xrtplot.XYCAxis(**y_kwds),
            caxis=xrtplot.XYCAxis(**e_kwds),
            aspect="auto",
            fluxKind="power",
        )
    )

    return result


def absorption(bl: SURFACE_VEIPS, plts: List):
    """"""
    bl.bm.eMin = 300
    bl.bm.eMax = 60000
    bl.bm.eN = 10001
    bl.CrlFocusApt.opening = [-0.5, 0.5, -0.5, 0.5]
    bl.CrlEntranceApt.opening = [-70, 70, -1.75, 1.75]

    del bl.CrocLensStack[:]
    bl.CrocLensStack = PrismaticLens.make_stack(
        L=90 * 3,
        N=30,  # 90 * 3,
        d=1.2 * 3,
        g_first=0.0,
        g_last=0.0,
        bl=bl,
        center=[0.0, 21.0e3, 0],
        material=rm.Be(kind="lens"),
        limPhysX=[-70, 70],
        limPhysY=[-5, 5],
    )

    for plot in plts:
        if plot.title == "CRL Focus":
            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None
        else:
            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None
    yield

    total_p = 0
    for plot in plts:
        if "BeamLensLocal2" in plot.title:
            print(plot.power)
            total_p += plot.power
    print(total_p)


# ################################## RUN ######################################
if __name__ == "__main__":
    beamline = SURFACE_VEIPS()
    scan = onept
    plot_gen = onept_plots
    show = False
    repeats = 10

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
