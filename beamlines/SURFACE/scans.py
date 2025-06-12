from typing import List

import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from branching_test import SURFACE


# ############################## ONEPT SCAN ###################################
def onept_plots(*args, **kwargs):
    """"""
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

    result = []
    for beam, t1 in zip(
        (
            "BeamSourceLocal",
            "BeamCRLEntranceLocal",
            "BeamCRLExitLocal",
            "BeamCRLFocusLocal",
        ),
        ("SRC", "CRL Entrance", "CRL Exit", "CRL Focus"),
    ):
        result.append(
            xrtplot.XYCPlot(
                beam=beam,
                title=t1,
                xaxis=xrtplot.XYCAxis(**x_kwds),
                yaxis=xrtplot.XYCAxis(**z_kwds),
                caxis=xrtplot.XYCAxis(**e_kwds),
                aspect="auto",
            )
        )

    return result


def onept(bl: SURFACE, plts: List):
    """"""
    bl.bm.eMin = 12.4e3
    bl.bm.eMax = 12.401e3
    # bl.CrlFocusApt.center = [0.0, 19990.0, 0.0]
    bl.CrlFocusApt.opening = [-120, 120, -0.2, 0.2]
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


# ################################## RUN ######################################
if __name__ == "__main__":
    beamline = SURFACE()
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
