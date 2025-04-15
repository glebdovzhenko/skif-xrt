# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-04-01"

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

be01 = rmatsel.Be(
    kind=r"lens")


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.undulator = rsources.Undulator(
        bl=beamLine,
        name=r"SCU",
        center=[0, 0, 0],
        eE=3,
        eI=0.4,
        eEspread=0.00135,
        eEpsilonX=0.096,
        eEpsilonZ=0.0096,
        betaX=15.66,
        betaZ=2.29,
        xPrimeMax=0.04,
        zPrimeMax=0.04,
        targetE=[8800, 3],
        eMin=8600,
        eMax=8900,
        period=15.6,
        n=128,
        targetOpenCL=[0, 0],
        precisionOpenCL=r"float32")

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        name=r"SourceMonitor",
        center=[0, 1000, 0])

    beamLine.roundAperture01 = rapts.RoundAperture(
        bl=beamLine,
        name=None,
        center=[0, 27990, 0])

    beamLine.doubleParaboloidLens01 = roes.DoubleParaboloidLens(
        bl=beamLine,
        name=None,
        center=[0, 28000, 0],
        pitch=1.5707963267948966,
        material=be01,
        limPhysX=[-1.0, 1.0],
        limPhysY=[-1.0, 1.0],
        shape=r"round",
        t=0.05,
        focus=0.5,
        zmax=0.5,
        nCRL=59,
        targetOpenCL=[0, 0],
        precisionOpenCL=r"float32")

    beamLine.screen02 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 35000, 0])

    beamLine.doubleParabolicCylinderLens01 = roes.DoubleParabolicCylinderLens(
        bl=beamLine,
        name=None,
        center=[0, 43000, 0],
        pitch=1.5707963267948966,
        material=be01,
        t=0.05,
        focus=0.5,
        zmax=0.5,
        nCRL=83,
        targetOpenCL=[2, 0],
        precisionOpenCL=r"float32")

    beamLine.screen03 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 51000, 0])

    return beamLine


def run_process(beamLine):
    undulator01beamGlobal01 = beamLine.undulator.shine()

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=undulator01beamGlobal01)

    roundAperture01beamLocal01 = beamLine.roundAperture01.propagate(
        beam=undulator01beamGlobal01)

    doubleParaboloidLens01beamGlobal01, doubleParaboloidLens01beamLocal101, doubleParaboloidLens01beamLocal201 = beamLine.doubleParaboloidLens01.multiple_refract(
        beam=undulator01beamGlobal01)

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=doubleParaboloidLens01beamGlobal01)

    doubleParabolicCylinderLens01beamGlobal01, doubleParabolicCylinderLens01beamLocal101, doubleParabolicCylinderLens01beamLocal201 = beamLine.doubleParabolicCylinderLens01.multiple_refract(
        beam=doubleParaboloidLens01beamGlobal01)

    screen03beamLocal01 = beamLine.screen03.expose(
        beam=doubleParabolicCylinderLens01beamGlobal01)

    outDict = {
        'undulator01beamGlobal01': undulator01beamGlobal01,
        'screen01beamLocal01': screen01beamLocal01,
        'roundAperture01beamLocal01': roundAperture01beamLocal01,
        'doubleParaboloidLens01beamGlobal01': doubleParaboloidLens01beamGlobal01,
        'doubleParaboloidLens01beamLocal101': doubleParaboloidLens01beamLocal101,
        'doubleParaboloidLens01beamLocal201': doubleParaboloidLens01beamLocal201,
        'screen02beamLocal01': screen02beamLocal01,
        'doubleParabolicCylinderLens01beamGlobal01': doubleParabolicCylinderLens01beamGlobal01,
        'doubleParabolicCylinderLens01beamLocal101': doubleParabolicCylinderLens01beamLocal101,
        'doubleParabolicCylinderLens01beamLocal201': doubleParabolicCylinderLens01beamLocal201,
        'screen03beamLocal01': screen03beamLocal01}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-0.15, 0.15],
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-0.15, 0.15],
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        title=r"Source XZ")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"roundAperture01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        title=r"Pinhole XZ")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"roundAperture01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x'",
            unit=r"",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z'",
            unit=r"",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        title=r"Pinhole X`Z`")
    plots.append(plot03)

    plot04 = xrtplot.XYCPlot(
        beam=r"doubleParaboloidLens01beamLocal201",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        title=r"Lens Exit XZ")
    plots.append(plot04)

    plot05 = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-0.15, 0.15],
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-0.15, 0.15],
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.2e"),
        title=r"Focus XZ")
    plots.append(plot05)
    return plots


def main():
    beamLine = build_beamline()
    E0 = 0.5 * (beamLine.undulator.eMin +
                beamLine.undulator.eMax)
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
