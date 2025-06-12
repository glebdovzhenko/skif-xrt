# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-06-12"

Created with xrtQook




"""

import numpy as np
import sys
sys.path.append(r"/nix/store/rywyrn2a3czklrlfzs1ly9f5sv7kyyjv-python3.12-xrt-1.6.1/lib/python3.12/site-packages")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials_elemental as rmatsel
import xrt.backends.raycing.materials_compounds as rmatsco
import xrt.backends.raycing.materials_crystals as rmatscr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.bendingMagnet01 = rsources.BendingMagnet(
        bl=beamLine,
        center=[0, 0, 0],
        eE=3,
        eI=0.4,
        eEspread=0.00135,
        eEpsilonX=0.075,
        eEpsilonZ=0.0075,
        betaX=15.66,
        betaZ=2.29,
        xPrimeMax=5,
        zPrimeMax=1,
        eMin=5000.0,
        eMax=15000.0,
        B0=2)

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 10000, 0])

    return beamLine


def run_process(beamLine):
    bendingMagnet01beamGlobal01 = beamLine.bendingMagnet01.shine()

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=bendingMagnet01beamGlobal01)

    outDict = {
        'bendingMagnet01beamGlobal01': bendingMagnet01beamGlobal01,
        'screen01beamLocal01': screen01beamLocal01}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        aspect=r"auto",
        title=r"plot01")
    plots.append(plot01)
    return plots


def main():
    beamLine = build_beamline()
    E0 = 0.5 * (beamLine.bendingMagnet01.eMin +
                beamLine.bendingMagnet01.eMax)
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
