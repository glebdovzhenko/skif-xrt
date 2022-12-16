# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2022-11-25"

Created with xrtQook




"""

import numpy as np
import sys
sys.path.append(r"/Users/glebdovzhenko/Dropbox/PycharmProjects/xrt")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.wiggler01 = rsources.Wiggler(
        bl=beamLine,
        center=[0, 0, 0])

    beamLine.rectangularAperture01 = rapts.RectangularAperture(
        bl=beamLine,
        center=[0, 15000, 0],
        opening=[-1, 1, -10, 10])

    return beamLine


def run_process(beamLine):
    wiggler01beamGlobal01 = beamLine.wiggler01.shine()

    rectangularAperture01beamLocal01 = beamLine.rectangularAperture01.propagate(
        beam=wiggler01beamGlobal01)

    outDict = {
        'wiggler01beamGlobal01': wiggler01beamGlobal01,
        'rectangularAperture01beamLocal01': rectangularAperture01beamLocal01}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"rectangularAperture01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            data=raycing.get_x,
            limits=[-1.5, 1.5]),
        yaxis=xrtplot.XYCAxis(
            label=r"x'",
            unit=r"",
            data=raycing.get_xprime,
            limits=[-0.0001, 0.0001]),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        aspect=r"auto",
        title=r"plot01")
    plots.append(plot01)
    return plots


def main():
    beamLine = build_beamline()
    E0 = 0.5 * (beamLine.wiggler01.eMin +
                beamLine.wiggler01.eMax)
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
