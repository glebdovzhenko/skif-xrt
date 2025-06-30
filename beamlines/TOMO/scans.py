from typing import List
import numpy as np
from matplotlib import pyplot as plt

import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun
import xrt.backends.raycing.materials_elemental as rm

from components import PrismaticLens
from tomo import LABTOMO


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
        ("BeamSampleLocal1", "BeamSampleLocal2", "BeamDetectorLocal"),
        (
            "Sample Entrance",
            "Sample Exit",
            "Detector",
        ),
        (
            y_kwds,
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


def onept(bl: LABTOMO, plts: List):
    """"""
    yield


# ################################## RUN ######################################
if __name__ == "__main__":
    beamline = LABTOMO()
    scan = onept
    plot_gen = onept_plots
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
