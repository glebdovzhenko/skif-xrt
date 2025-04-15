# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-04-15"

Created with xrtQook




"""

import numpy as np
import sys
sys.path.append(r"D:\miniconda3\Lib\site-packages")
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

    beamLine.undulator01 = rsources.Undulator(
        bl=beamLine,
        center=[0, 0, 0],
        eE=3.0,
        eI=0.4,
        eEspread=0.00135,
        eEpsilonX=0.15,
        eEpsilonZ=0.015,
        betaX=15.66,
        betaZ=2.29,
        xPrimeMax=0.05,
        zPrimeMax=0.05,
        targetE=[8800, 3],
        eMin=8799.5,
        eMax=8800.5,
        eN=101,
        period=15.6,
        n=128,
        targetOpenCL=[0, 0],
        precisionOpenCL=r"auto")

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 28000, 0])

    beamLine.screen02 = rscreens.Screen(
        bl=beamLine,
        name=None)

    return beamLine


def run_process(beamLine):
    undulator01beamGlobal01 = beamLine.undulator01.shine()

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=undulator01beamGlobal01)

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=undulator01beamGlobal01)

    outDict = {
        'undulator01beamGlobal01': undulator01beamGlobal01,
        'screen01beamLocal01': screen01beamLocal01,
        'screen02beamLocal01': screen02beamLocal01}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x'",
            unit=r"",
            bins=512,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"z'",
            unit=r"",
            bins=512,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        title=r"Directions")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x'",
            unit=r"",
            limits=[-0.0001, 0.0001],
            bins=512,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        aspect=r"auto",
        title=r"$x^{\prime}(E)$")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"z'",
            unit=r"",
            limits=[-0.0001, 0.0001],
            bins=512,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        aspect=r"auto",
        title=r"$Z^{\prime}(E)$")
    plots.append(plot03)

    plot04 = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-0.15, 0.15],
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-0.015, 0.015],
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        aspect=r"auto",
        title=r"Source XZ")
    plots.append(plot04)
    return plots


def main():
    beamLine = build_beamline()
    E0 = 0.5 * (beamLine.undulator01.eMin +
                beamLine.undulator01.eMax)
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=10,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
