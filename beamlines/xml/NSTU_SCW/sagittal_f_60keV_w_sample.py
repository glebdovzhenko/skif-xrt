# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-04-19"

Created with xrtQook




"""

import numpy as np
import sys
sys.path.append(r"/nix/store/zms2fc9xb0cgjnbixx3szmqnqwf480f5-python3.12-xrt-1.6.1/lib/python3.12/site-packages")
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

crystalSi01 = rmats.CrystalSi(
    t=0.5,
    geom=r"Laue reflected",
    name=None,
    volumetricDiffraction=True,
    useTT=True)

powder01 = rmats.Powder(
    chi=[0, 6.283185307179586],
    name=None,
    hkl=[7, 7, 7],
    a=5.256,
    atoms=[58, 58, 58, 58, 8, 8, 8, 8, 8, 8, 8, 8],
    atomsXYZ=[[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25], [0.75, 0.75, 0.75], [0.75, 0.25, 0.25], [0.25, 0.75, 0.25], [0.25, 0.25, 0.75]],
    tK=297.15,
    t=1)


def build_beamline():
    beamLine = raycing.BeamLine(
        alignE=60000.0)

    beamLine.wiggler01 = rsources.Wiggler(
        bl=beamLine,
        center=[0, 0, 0],
        nrays=1000000,
        eE=3.0,
        eI=0.4,
        eEspread=0.00135,
        eEpsilonX=0.075,
        eEpsilonZ=0.0075,
        betaX=15.66,
        betaZ=2.29,
        xPrimeMax=1,
        zPrimeMax=0.1,
        eMin=59500,
        eMax=60500,
        eN=101,
        K=20.1685,
        period=48.0,
        n=18)

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 20000, 0])

    beamLine.bentLaue2D01 = roes.BentLaue2D(
        bl=beamLine,
        name=None,
        center=[0, 33000, 0],
        pitch=2.21984,
        material=crystalSi01,
        alpha=0.6161012259539983,
        Rm=6144.0,
        Rs=-1024,
        targetOpenCL=[2, 0])

    beamLine.screen02 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 33050, 0])

    beamLine.bentLaue2D02 = roes.BentLaue2D(
        bl=beamLine,
        name=None,
        center=[0, 33378.803, 25],
        pitch=2.153939,
        positionRoll=3.141592653589793,
        material=crystalSi01,
        alpha=0.6161012259539983,
        Rm=6144.0,
        Rs=-1024,
        targetOpenCL=[2, 0])

    beamLine.rectangularAperture01 = rapts.RectangularAperture(
        bl=beamLine,
        name=None,
        center=[0, 55990, 25],
        opening=[-0.068, 0.068, -0.69, 0.69])

    beamLine.lauePlate01 = roes.LauePlate(
        bl=beamLine,
        name=None,
        center=[0, 56000, 25],
        pitch=1.5707963267948966,
        material=powder01,
        targetOpenCL=[0, 0],
        precisionOpenCL=r"float32")

    beamLine.roundBeamStop01 = rapts.RoundBeamStop(
        bl=beamLine,
        name=None,
        center=[0, 56499, 25])

    beamLine.screen03 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 56500, 25])

    return beamLine


def run_process(beamLine):
    wiggler01beamGlobal01 = beamLine.wiggler01.shine()

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=wiggler01beamGlobal01)

    bentLaue2D01beamGlobal01, bentLaue2D01beamLocal01 = beamLine.bentLaue2D01.reflect(
        beam=wiggler01beamGlobal01)

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=bentLaue2D01beamGlobal01)

    bentLaue2D02beamGlobal01, bentLaue2D02beamLocal01 = beamLine.bentLaue2D02.reflect(
        beam=bentLaue2D01beamGlobal01)

    rectangularAperture01beamLocal01 = beamLine.rectangularAperture01.propagate(
        beam=bentLaue2D02beamGlobal01)

    lauePlate01beamGlobal01, lauePlate01beamLocal01 = beamLine.lauePlate01.reflect(
        beam=bentLaue2D02beamGlobal01)

    roundBeamStop01beamLocal01 = beamLine.roundBeamStop01.propagate(
        beam=lauePlate01beamGlobal01)

    screen03beamLocal01 = beamLine.screen03.expose(
        beam=lauePlate01beamGlobal01)

    outDict = {
        'wiggler01beamGlobal01': wiggler01beamGlobal01,
        'screen01beamLocal01': screen01beamLocal01,
        'bentLaue2D01beamGlobal01': bentLaue2D01beamGlobal01,
        'bentLaue2D01beamLocal01': bentLaue2D01beamLocal01,
        'screen02beamLocal01': screen02beamLocal01,
        'bentLaue2D02beamGlobal01': bentLaue2D02beamGlobal01,
        'bentLaue2D02beamLocal01': bentLaue2D02beamLocal01,
        'rectangularAperture01beamLocal01': rectangularAperture01beamLocal01,
        'lauePlate01beamGlobal01': lauePlate01beamGlobal01,
        'lauePlate01beamLocal01': lauePlate01beamLocal01,
        'roundBeamStop01beamLocal01': roundBeamStop01beamLocal01,
        'screen03beamLocal01': screen03beamLocal01}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"z"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        aspect=r"auto",
        title=r"FRONT-END")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"bentLaue2D01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=256,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            bins=256,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=256,
            ppb=1),
        aspect=r"auto",
        title=r"CRYSTAL 1")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"bentLaue2D02beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=256,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            bins=256,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=256,
            ppb=1),
        aspect=r"auto",
        title=r"CRYSTAL 2")
    plots.append(plot03)

    plot04 = xrtplot.XYCPlot(
        beam=r"rectangularAperture01beamLocal01",
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
            ppb=1),
        aspect=r"auto",
        title=r"EXIT")
    plots.append(plot04)

    plot05 = xrtplot.XYCPlot(
        beam=r"screen03beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-100, 100],
            bins=1000,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-100, 100],
            bins=1000,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=256,
            ppb=1),
        title=r"DETECTOR")
    plots.append(plot05)
    return plots


def main():
    beamLine = build_beamline()
    E0 = 0.5 * (beamLine.wiggler01.eMin +
                beamLine.wiggler01.eMax)
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=10,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
