from typing import List
import numpy as np
from matplotlib import pyplot as plt

import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun
import xrt.backends.raycing.materials_elemental as rm

from components import PrismaticLens
from surface_base import SURFACE_BASE


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
            "BeamCRLFocusLocal",
        ),
        (
            "SRC",
            "CRL Entrance",
            "CRL Exit",
            "CRL Focus",
        ),
        (
            z_kwds,
            z_kwds,
            z_kwds,
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


def onept(bl: SURFACE_BASE, plts: List):
    """"""
    bl.bm.eMin = 10e3 * (1.0 - 5e-4)
    bl.bm.eMax = 10e3 * (1.0 + 5e-4)
    bl.CrlFocusApt.opening = [-120, 120, -120, 120]
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


# ############################ LENS FOCUS SCAN ################################
def lens_focus_plots(*args, **kwargs):
    """"""
    result = []
    for beam, t1, y_k in zip(
        (
            "BeamSourceLocal",
            "BeamCRLEntranceLocal",
            "BeamCRLExitLocal",
            "BeamCRLFocusLocal",
        ),
        (
            "SRC",
            "CRL Entrance",
            "CRL Exit",
            "CRL Focus",
        ),
        (
            z_kwds,
            z_kwds,
            z_kwds,
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


def lens_focus(bl: SURFACE_BASE, plts: List):
    bl.bm.eMin = 12.4e3 * (1.0 - 5e-4)
    bl.bm.eMax = 12.4e3 * (1.0 + 5e-4)
    bl.CrlFocusApt.opening = [-120, 120, -0.2, 0.2]

    result = []
    for y_g in np.linspace(1.730, 1.770, 20):
        del bl.CrocLensStack[:]
        bl.CrocLensStack = PrismaticLens.make_stack(
            L=90 * 3,
            N=90 * 3,
            # N=500,
            d=1.2 * 3,
            g_first=y_g,
            g_last=0.0,
            bl=bl,
            center=[0.0, 21e3, 0],
            material=rm.Be(kind="lens"),
            limPhysX=[-70, 70],
            limPhysY=[-5, 5],
        )
        bl.CrlEntranceApt.opening = [-70, 70, -y_g, y_g]

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

        for plot in plts:
            if plot.title == "CRL Focus":
                print(f"Image {plot.dy}")
                result.append([y_g, plot.dy])

    result = np.array(result)
    print(result)
    plt.figure()
    plt.plot(*(result.T))
    plt.show()


# ########################### LENS APERTURE SCAN ##############################
def lens_apt_plots(*args, **kwargs):
    """"""
    result = []
    for beam, t1, y_k in zip(
        (
            "BeamSourceLocal",
            "BeamCRLEntranceLocal",
            "BeamCRLExitLocal",
            "BeamCRLFocusLocal",
        ),
        (
            "SRC",
            "CRL Entrance",
            "CRL Exit",
            "CRL Focus",
        ),
        (
            z_kwds,
            z_kwds,
            z_kwds,
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


def lens_apt(bl: SURFACE_BASE, plts: List):
    bl.bm.eMin = 12.4e3 * (1.0 - 5e-4)
    bl.bm.eMax = 12.4e3 * (1.0 + 5e-4)
    bl.CrlFocusApt.opening = [-120, 120, -0.2, 0.2]

    del bl.CrocLensStack[:]
    bl.CrocLensStack = PrismaticLens.make_stack(
        L=90 * 3,
        N=90 * 3,
        d=1.2 * 3,
        g_first=1.751,
        g_last=0.0,
        bl=bl,
        center=[0.0, 21e3, 0],
        material=rm.Be(kind="lens"),
        limPhysX=[-70, 70],
        limPhysY=[-5, 5],
    )

    result = []
    flux, dz = [], []
    for y_g in np.linspace(0.05, 0.8, 20):
        bl.CrlEntranceApt.opening = [-70, 70, -y_g, y_g]

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

        for plot in plts:
            if plot.title == "CRL Focus":
                print(f"Image {plot.flux, plot.dy}")
                flux.append([y_g, plot.flux])
                dz.append([y_g, plot.dy])

    flux = np.array(flux)
    dz = np.array(dz)
    plt.figure()
    plt.plot(*(flux.T))
    plt.figure()
    plt.plot(*(dz.T))
    plt.show()


# ############################# LENS TEETH SCAN ###############################
def lens_teeth_plots(*args, **kwargs):
    """"""
    result = []
    for beam, t1, y_k in zip(
        (
            "BeamSourceLocal",
            "BeamCRLEntranceLocal",
            "BeamCRLExitLocal",
            "BeamCRLFocusLocal",
        ),
        (
            "SRC",
            "CRL Entrance",
            "CRL Exit",
            "CRL Focus",
        ),
        (
            z_kwds,
            z_kwds,
            z_kwds,
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


def lens_teeth(bl: SURFACE_BASE, plts: List):
    bl.bm.eMin = 12.4e3 * (1.0 - 5e-4)
    bl.bm.eMax = 12.4e3 * (1.0 + 5e-4)
    bl.CrlFocusApt.opening = [-120, 120, -0.2, 0.2]

    flux, dz = [], []
    for nteeth in [50, 100, 150, 200, 250, 300, 350, 400, 450]:

        del bl.CrocLensStack[:]
        bl.CrocLensStack = PrismaticLens.make_stack(
            L=90 * 3,
            N=nteeth,
            d=1.2 * 3,
            g_first=1.751,
            g_last=0.0,
            bl=bl,
            center=[0.0, 21e3, 0],
            material=rm.Be(kind="lens"),
            limPhysX=[-70, 70],
            limPhysY=[-5, 5],
        )

        bl.CrlEntranceApt.opening = [-70, 70, -1.751, 1.75]

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

        for plot in plts:
            if plot.title == "CRL Focus":
                print(f"Image {plot.flux, plot.dy}")
                flux.append([nteeth, plot.flux])
                dz.append([nteeth, plot.dy])

    flux = np.array(flux)
    dz = np.array(dz)
    plt.figure()
    plt.plot(*(flux.T))
    plt.figure()
    plt.plot(*(dz.T))
    plt.show()


# ################################## RUN ######################################
if __name__ == "__main__":
    beamline = SURFACE_BASE()
    scan = onept
    plot_gen = onept_plots
    show = False
    repeats = 1

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
