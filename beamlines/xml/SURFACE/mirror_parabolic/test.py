# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-08-13"

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

si01 = rmatsel.Si(
    kind=r"mirror")

pt01 = rmatsel.Pt(
    kind=r"mirror")

coated01 = rmats.Coated(
    coating=pt01,
    cThickness=100,
    substrate=si01,
    name=None,
    geom=r"")

rh01 = rmatsel.Rh()

coated02 = rmats.Coated(
    coating=rh01,
    cThickness=100,
    substrate=si01,
    geom=r"")


def build_beamline():
    beamLine = raycing.BeamLine(
        alignE=15000)

    beamLine.bendingMagnet01 = rsources.BendingMagnet(
        bl=beamLine,
        name=r"Source",
        center=[0, 0, 0],
        nrays=1000000,
        eE=3,
        eI=0.4,
        eEspread=0.00135,
        eEpsilonX=0.075,
        eEpsilonZ=0.0075,
        betaX=0.252,
        betaZ=7.77,
        xPrimeMax=2.5,
        zPrimeMax=2.0,
        eMin=5000,
        eMax=15000)

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        name=None)

    beamLine.screen02 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 20000, 0])

    beamLine.bentFlatMirror01 = roes.BentFlatMirror(
        bl=beamLine,
        name=None,
        center=[0, 21000, 0],
        pitch=0.003490658503988659,
        material=coated01,
        R=6000000.0,
        targetOpenCL=r"GPU",
        precisionOpenCL=r"float32")

    beamLine.screen03 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 22000, 0])

    beamLine.screen04 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 42000, r"auto"])

    return beamLine


def run_process(beamLine):
    bendingMagnet01beamGlobal01 = beamLine.bendingMagnet01.shine()

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=bendingMagnet01beamGlobal01)

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=bendingMagnet01beamGlobal01)

    bentFlatMirror01beamGlobal01, bentFlatMirror01beamLocal01 = beamLine.bentFlatMirror01.reflect(
        beam=bendingMagnet01beamGlobal01)

    screen03beamLocal01 = beamLine.screen03.expose(
        beam=bentFlatMirror01beamGlobal01)

    screen04beamLocal01 = beamLine.screen04.expose(
        beam=bentFlatMirror01beamGlobal01)

    outDict = {
        'bendingMagnet01beamGlobal01': bendingMagnet01beamGlobal01,
        'screen01beamLocal01': screen01beamLocal01,
        'screen02beamLocal01': screen02beamLocal01,
        'bentFlatMirror01beamGlobal01': bentFlatMirror01beamGlobal01,
        'bentFlatMirror01beamLocal01': bentFlatMirror01beamLocal01,
        'screen03beamLocal01': screen03beamLocal01,
        'screen04beamLocal01': screen04beamLocal01}
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
        title=r"Source XZ")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01",
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
        title=r"Pre mirror XZ")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"screen03beamLocal01",
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
        title=r"Post mirror XZ")
    plots.append(plot03)

    plot04 = xrtplot.XYCPlot(
        beam=r"screen04beamLocal01",
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
        title=r"Sample XZ")
    plots.append(plot04)
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
