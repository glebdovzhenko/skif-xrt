# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-07-09"

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

be01 = rmatsel.Be(
    kind=r"lens")

crystalSi01 = rmats.CrystalSi(
    name=None,
    useTT=True)


def build_beamline():
    beamLine = raycing.BeamLine(
        alignE=5000)

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
        eMin=4990,
        eMax=5010,
        B0=2)

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        name=r"Source Screen")

    beamLine.plate01 = roes.Plate(
        bl=beamLine,
        name=r"Filter 1",
        center=[0, 11000, 0],
        pitch=1.5707963267948966,
        material=be01,
        t=0.3,
        targetOpenCL=r"CPU",
        precisionOpenCL=r"float32")

    beamLine.plate02 = roes.Plate(
        bl=beamLine,
        name=r"Filter 1",
        center=[0, 13000, 0],
        pitch=1.5707963267948966,
        material=be01,
        t=0.3,
        targetOpenCL=r"CPU",
        precisionOpenCL=r"float32")

    beamLine.parabolicCylinderFlatLens01 = roes.ParabolicCylinderFlatLens(
        bl=beamLine,
        name=None,
        center=[0, 21000, 0],
        material=be01,
        focus=0.0973,
        zmax=270,
        targetOpenCL=r"CPU",
        precisionOpenCL=r"float32")

    beamLine.rectangularAperture01 = rapts.RectangularAperture(
        bl=beamLine,
        name=r"Splitter",
        center=[0, 22000, 0],
        opening=[-55, -11, -2, 2])

    beamLine.dcMwithSagittalFocusing01 = roes.DCMwithSagittalFocusing(
        bl=beamLine,
        name=r"Mono",
        center=[r"auto", 24000, 0],
        bragg=r"auto",
        yaw=0.0015,
        material=crystalSi01,
        material2=crystalSi01,
        limPhysX=[-70, 70],
        limPhysY=[-30, 30],
        Rs=8135,
        fixedOffset=25,
        targetOpenCL=r"CPU",
        precisionOpenCL=r"float32")

    beamLine.screen02 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[r"auto", 42000, 25])

    return beamLine


def run_process(beamLine):
    bendingMagnet01beamGlobal01 = beamLine.bendingMagnet01.shine()

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=bendingMagnet01beamGlobal01)

    plate01beamGlobal01, plate01beamLocal101, plate01beamLocal201 = beamLine.plate01.double_refract(
        beam=bendingMagnet01beamGlobal01,
        returnLocalAbsorbed=0)

    plate02beamGlobal01, plate02beamLocal101, plate02beamLocal201 = beamLine.plate02.double_refract(
        beam=plate01beamGlobal01,
        returnLocalAbsorbed=0)

    parabolicCylinderFlatLens01beamGlobal01, parabolicCylinderFlatLens01beamLocal101, parabolicCylinderFlatLens01beamLocal201 = beamLine.parabolicCylinderFlatLens01.multiple_refract(
        beam=plate02beamGlobal01)

    rectangularAperture01beamLocal01 = beamLine.rectangularAperture01.propagate(
        beam=parabolicCylinderFlatLens01beamGlobal01)

    dcMwithSagittalFocusing01beamGlobal01, dcMwithSagittalFocusing01beamLocal101, dcMwithSagittalFocusing01beamLocal201 = beamLine.dcMwithSagittalFocusing01.double_reflect(
        beam=parabolicCylinderFlatLens01beamGlobal01)

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=dcMwithSagittalFocusing01beamGlobal01)

    outDict = {
        'bendingMagnet01beamGlobal01': bendingMagnet01beamGlobal01,
        'screen01beamLocal01': screen01beamLocal01,
        'plate01beamGlobal01': plate01beamGlobal01,
        'plate01beamLocal101': plate01beamLocal101,
        'plate01beamLocal201': plate01beamLocal201,
        'plate02beamGlobal01': plate02beamGlobal01,
        'plate02beamLocal101': plate02beamLocal101,
        'plate02beamLocal201': plate02beamLocal201,
        'parabolicCylinderFlatLens01beamGlobal01': parabolicCylinderFlatLens01beamGlobal01,
        'parabolicCylinderFlatLens01beamLocal101': parabolicCylinderFlatLens01beamLocal101,
        'parabolicCylinderFlatLens01beamLocal201': parabolicCylinderFlatLens01beamLocal201,
        'rectangularAperture01beamLocal01': rectangularAperture01beamLocal01,
        'dcMwithSagittalFocusing01beamGlobal01': dcMwithSagittalFocusing01beamGlobal01,
        'dcMwithSagittalFocusing01beamLocal101': dcMwithSagittalFocusing01beamLocal101,
        'dcMwithSagittalFocusing01beamLocal201': dcMwithSagittalFocusing01beamLocal201,
        'screen02beamLocal01': screen02beamLocal01}
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
        beam=r"plate01beamLocal201",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
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
        title=r"Filter 1 Abs",
        fluxKind=r"power")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"plate02beamLocal201",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
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
        title=r"Filter 2 Abs",
        fluxKind=r"power")
    plots.append(plot03)

    plot04 = xrtplot.XYCPlot(
        beam=r"dcMwithSagittalFocusing01beamLocal101",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
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
        title=r"DCM C1")
    plots.append(plot04)

    plot05 = xrtplot.XYCPlot(
        beam=r"dcMwithSagittalFocusing01beamLocal201",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
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
        title=r"DCM C2")
    plots.append(plot05)

    plot06 = xrtplot.XYCPlot(
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
        title=r"Sample")
    plots.append(plot06)
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
