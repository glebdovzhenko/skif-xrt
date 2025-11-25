# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-08-11"

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

crystalSi01 = rmats.CrystalSi(
    name=None,
    useTT=True)

si01 = rmatsel.Si()

pt01 = rmatsel.Pt()

coated01 = rmats.Coated(
    coating=pt01,
    cThickness=100,
    substrate=si01,
    name=None,
    geom=r"")


def build_beamline():
    beamLine = raycing.BeamLine(
        alignE=12400)

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
        eMin=12380,
        eMax=12420,
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

    beamLine.bentFlatMirror01 = roes.BentFlatMirror(
        bl=beamLine,
        name=None,
        center=[0, 21000, 0],
        pitch=0.003490658503988659,
        material=coated01,
        limPhysX=[-60.0, 60.0],
        limPhysY=[-200.0, 200.0],
        R=5800000.0,
        targetOpenCL=r"CPU",
        precisionOpenCL=r"float32")

    beamLine.rectangularAperture01 = rapts.RectangularAperture(
        bl=beamLine,
        name=r"Splitter",
        center=[0, 22000, 7],
        opening=[11, 55, -2, 2])

    beamLine.johannCylinder01 = roes.JohannCylinder(
        bl=beamLine,
        name=r"Monochromtaor",
        center=[r"auto", 24000, r"auto"],
        pitch=r"auto",
        roll=1.5707963267948966,
        yaw=0.003490658503988659,
        material=crystalSi01,
        limPhysX=[-2.0, 2.0],
        limPhysY=[-150.0, 150.0],
        Rm=124000,
        targetOpenCL=r"CPU",
        precisionOpenCL=r"float32")

    beamLine.screen02 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[r"auto", 40000, r"auto"])

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

    bentFlatMirror01beamGlobal01, bentFlatMirror01beamLocal01 = beamLine.bentFlatMirror01.reflect(
        beam=plate02beamGlobal01)

    rectangularAperture01beamLocal01 = beamLine.rectangularAperture01.propagate(
        beam=bentFlatMirror01beamGlobal01)

    johannCylinder01beamGlobal01, johannCylinder01beamLocal01 = beamLine.johannCylinder01.reflect(
        beam=bentFlatMirror01beamGlobal01)

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=johannCylinder01beamGlobal01)

    outDict = {
        'bendingMagnet01beamGlobal01': bendingMagnet01beamGlobal01,
        'screen01beamLocal01': screen01beamLocal01,
        'plate01beamGlobal01': plate01beamGlobal01,
        'plate01beamLocal101': plate01beamLocal101,
        'plate01beamLocal201': plate01beamLocal201,
        'plate02beamGlobal01': plate02beamGlobal01,
        'plate02beamLocal101': plate02beamLocal101,
        'plate02beamLocal201': plate02beamLocal201,
        'bentFlatMirror01beamGlobal01': bentFlatMirror01beamGlobal01,
        'bentFlatMirror01beamLocal01': bentFlatMirror01beamLocal01,
        'rectangularAperture01beamLocal01': rectangularAperture01beamLocal01,
        'johannCylinder01beamGlobal01': johannCylinder01beamGlobal01,
        'johannCylinder01beamLocal01': johannCylinder01beamLocal01,
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
        title=r"Sample",
        saveName=r"sample_veips_12.4.png")
    plots.append(plot04)

    plot05 = xrtplot.XYCPlot(
        beam=r"johannCylinder01beamLocal01",
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
        title=r"Crystal Footprint")
    plots.append(plot05)
    return plots


def main():
    beamLine = build_beamline()
    E0 = 0.5 * (beamLine.bendingMagnet01.eMin +
                beamLine.bendingMagnet01.eMax)
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=5,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
