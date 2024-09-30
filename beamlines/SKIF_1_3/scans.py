from typing import List
import numpy as np
import os
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from SKIF_1_3 import SKIF13


# ################################### PLOTS ###################################


plots = [
    xrtplot.XYCPlot(
        beam="WgMonitorLocal",
        title="Wiggler Monitor",
        xaxis=xrtplot.XYCAxis(r"$x$", "mm", limits=(0.0, 0.0)),
        yaxis=xrtplot.XYCAxis(r"$z$", "mm", limits=(0.0, 0.0)),
        aspect="auto",
    ),
    xrtplot.XYCPlot(
        beam="FEMonitorLocal",
        title="Front-End Monitor",
        xaxis=xrtplot.XYCAxis(r"$x$", "mm", limits=(0.0, 0.0)),
        yaxis=xrtplot.XYCAxis(r"$z$", "mm", limits=(0.0, 0.0)),
        aspect="auto",
    ),
    xrtplot.XYCPlot(
        beam="LensEntranceMonitorLocal",
        title="Lens Entrance Monitor",
        xaxis=xrtplot.XYCAxis(r"$x$", "mm", limits=(0.0, 0.0)),
        yaxis=xrtplot.XYCAxis(r"$z$", "mm", limits=(0.0, 0.0)),
        aspect="auto",
    ),
    xrtplot.XYCPlot(
        beam="LensExitMonitorLocal",
        title="Lens Exit Monitor",
        xaxis=xrtplot.XYCAxis(r"$x$", "mm", limits=(0.0, 0.0)),
        yaxis=xrtplot.XYCAxis(r"$z$", "mm", limits=(0.0, 0.0)),
        aspect="auto",
    ),
    xrtplot.XYCPlot(
        beam="SampleMonitorLocal",
        title="Sample Monitor",
        xaxis=xrtplot.XYCAxis(r"$x$", "mm", limits=(0.0, 0.0)),
        yaxis=xrtplot.XYCAxis(r"$z$", "mm", limits=(0.0, 0.0)),
        aspect="auto",
    ),
]


def setup_plots(plts: List[xrtplot.XYCPlot], bl: SKIF13):
    # setting up axes limits
    for plt in plts:
        if plt.beam in ("WgMonitorLocal", "FEMonitorLocal"):
            plt.xaxis.limits = (
                1.4 * bl.FrontEnd.opening[0],
                1.4 * bl.FrontEnd.opening[1],
            )
            plt.yaxis.limits = (
                1.4 * bl.FrontEnd.opening[2],
                1.4 * bl.FrontEnd.opening[3],
            )
        if plt.beam in ("LensEntranceMonitorLocal", "LensExitMonitorLocal"):
            plt.xaxis.limits = (
                1.4 * bl.CrocLensStack[0].limPhysX[0],
                1.4 * bl.CrocLensStack[0].limPhysX[1],
            )
            plt.yaxis.limits = (
                1.4 * bl.CrocLensStack[0].limPhysY[0],
                1.4 * bl.CrocLensStack[0].limPhysY[1],
            )
            if plt.beam == "LensExitMonitorLocal":
                plt.yaxis.limits = (
                    -0.7 * bl.lens_pars["Aperture"],
                    0.7 * bl.lens_pars["Aperture"],
                )
        if plt.beam == "SampleMonitorLocal":
            plt.xaxis.limits = (
                1.4 * bl.SampleSlit.opening[0],
                1.4 * bl.SampleSlit.opening[1],
            )
            plt.yaxis.limits = (
                -0.7 * bl.lens_pars["Aperture"],
                0.7 * bl.lens_pars["Aperture"],
            )

    # setting up plot titles
    for plt in plts:
        if plt.beam == "WgMonitorLocal":
            plt.title = " ".join(
                (plt.title, "[%.01f m]" % (bl.WigglerMonitor.center[1] * 1e-3))
            )
        elif plt.beam == "FEMonitorLocal":
            plt.title = " ".join(
                (plt.title, "[%.01f m]" % (bl.FrontEndMonitor.center[1] * 1e-3))
            )
        elif plt.beam == "LensEntranceMonitorLocal":
            plt.title = " ".join(
                (plt.title, "[%.01f m]" % (bl.LensEntranceMonitor.center[1] * 1e-3))
            )
        elif plt.beam == "LensExitMonitorLocal":
            plt.title = " ".join(
                (plt.title, "[%.01f m]" % (bl.LensExitMonitor.center[1] * 1e-3))
            )
        elif plt.beam == "SampleMonitorLocal":
            plt.title = " ".join(
                (plt.title, "[%.01f m]" % (bl.SampleMonitor.center[1] * 1e-3))
            )


# ################################### SCANS ###################################


def none_scan(plts: List[xrtplot.XYCPlot], bl: SKIF13):
    setup_plots(plts, bl)
    yield


def en_scan(plts: List[xrtplot.XYCPlot], bl: SKIF13):
    subdir = os.path.join(os.getenv("BASE_DIR", ""), "datasets", "skif13")
    scan_name = "1mm_gr_lens"
    if not os.path.exists(os.path.join(subdir, scan_name)):
        os.mkdir(os.path.join(subdir, scan_name))

    setup_plots(plts, bl)
    for plt in plts:
        if plt.beam == "SampleMonitorLocal":
            plt.xaxis.limits = (
                1.4 * bl.SampleSlit.opening[0],
                1.4 * bl.SampleSlit.opening[1],
            )
            plt.yaxis.limits = (
                1.4 * bl.SampleSlit.opening[2],
                1.4 * bl.SampleSlit.opening[3],
            )
    ens = np.concatenate(
        (
            np.array([31.0, 32.0, 33.0, 34.0, 36.0, 37.0, 38.0, 39.0]),
            np.arange(10.0, 70.0, 5.0),
        )
    )
    for en in ens * 1e3:
        bl.SuperCWiggler.eMin = en - 1.0
        bl.SuperCWiggler.eMax = en + 1.0
        for plt in plts:
            plt.saveName = " ".join((plt.title, "[%.01f keV]" % (en * 1e-3)))
            plt.saveName = os.path.join(subdir, scan_name, plt.saveName + ".png")
            plt.persistentName = plt.saveName.replace(".png", ".pickle")
            plt.caxis.limits = None
            # if plt.beam == "SampleMonitorLocal":
            #     plt.yaxis.limits = None
        yield


# ################################### MAIN ####################################


if __name__ == "__main__":
    beamline = SKIF13()
    scan = none_scan
    show = False
    repeats = 1

    if show:
        beamline.glow(
            scale=[1e3, 1e3, 1e3],
            centerAt=2,
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
