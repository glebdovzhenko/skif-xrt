# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-11-14"

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

rh01 = rmatsel.Rh()

coated02 = rmats.Coated(
    coating=rh01,
    cThickness=100,
    substrate=si01,
    name=None,
    geom=r"")

crystalHarmonics01 = rmats.CrystalHarmonics()


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
        xPrimeMax=2.0,
        zPrimeMax=0.2,
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

    beamLine.parabolicalMirrorParam01 = roes.ParabolicalMirrorParam(
        bl=beamLine,
        name=None,
        center=[0, 22000, 0],
        pitch=0.001,
        material=coated02,
        limPhysX=[-60.0, 60.0],
        limPhysY=[-600.0, 600.0],
        p=22000,
        q=None,
        isCylindrical=True)

    beamLine.screen02 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 23000, 0])

    beamLine.dcMwithSagittalFocusing01 = roes.DCMwithSagittalFocusing(
        bl=beamLine,
        name=None,
        center=[0, 24000, r"auto"],
        bragg=r"auto",
        pitch=0.002,
        material=crystalHarmonics01,
        material2=crystalHarmonics01,
        Rs=9400.0,
        fixedOffset=25,
        targetOpenCL=[2, 0])

    beamLine.screen03 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 25000, r"auto"])

    beamLine.parabolicalMirrorParam02 = roes.ParabolicalMirrorParam(
        bl=beamLine,
        name=None,
        center=[0, 26000, r"auto"],
        pitch=3.142592653589793,
        material=coated02,
        limPhysX=[-60.0, 60.0],
        limPhysY=[-600.0, 600.0],
        p=22000,
        q=None,
        isCylindrical=True,
        targetOpenCL=[2, 0])

    beamLine.screen04 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 48000, r"auto"])

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

    parabolicalMirrorParam01beamGlobal01, parabolicalMirrorParam01beamLocal01 = beamLine.parabolicalMirrorParam01.reflect(
        beam=plate02beamGlobal01)

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=parabolicalMirrorParam01beamGlobal01)

    dcMwithSagittalFocusing01beamGlobal01, dcMwithSagittalFocusing01beamLocal101, dcMwithSagittalFocusing01beamLocal201 = beamLine.dcMwithSagittalFocusing01.double_reflect(
        beam=parabolicalMirrorParam01beamGlobal01)

    screen03beamLocal01 = beamLine.screen03.expose(
        beam=dcMwithSagittalFocusing01beamGlobal01)

    parabolicalMirrorParam02beamGlobal01, parabolicalMirrorParam02beamLocal01 = beamLine.parabolicalMirrorParam02.reflect(
        beam=dcMwithSagittalFocusing01beamGlobal01)

    screen04beamLocal01 = beamLine.screen04.expose(
        beam=parabolicalMirrorParam02beamGlobal01)

    outDict = {
        'bendingMagnet01beamGlobal01': bendingMagnet01beamGlobal01,
        'screen01beamLocal01': screen01beamLocal01,
        'plate01beamGlobal01': plate01beamGlobal01,
        'plate01beamLocal101': plate01beamLocal101,
        'plate01beamLocal201': plate01beamLocal201,
        'plate02beamGlobal01': plate02beamGlobal01,
        'plate02beamLocal101': plate02beamLocal101,
        'plate02beamLocal201': plate02beamLocal201,
        'parabolicalMirrorParam01beamGlobal01': parabolicalMirrorParam01beamGlobal01,
        'parabolicalMirrorParam01beamLocal01': parabolicalMirrorParam01beamLocal01,
        'screen02beamLocal01': screen02beamLocal01,
        'dcMwithSagittalFocusing01beamGlobal01': dcMwithSagittalFocusing01beamGlobal01,
        'dcMwithSagittalFocusing01beamLocal101': dcMwithSagittalFocusing01beamLocal101,
        'dcMwithSagittalFocusing01beamLocal201': dcMwithSagittalFocusing01beamLocal201,
        'screen03beamLocal01': screen03beamLocal01,
        'parabolicalMirrorParam02beamGlobal01': parabolicalMirrorParam02beamGlobal01,
        'parabolicalMirrorParam02beamLocal01': parabolicalMirrorParam02beamLocal01,
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
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x'",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z'",
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
        title=r"Source X'Z'")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x'",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z'",
            offset=0.002,
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
        title=r"Mirror X'Z'")
    plots.append(plot03)

    plot05 = xrtplot.XYCPlot(
        beam=r"screen03beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x'",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z'",
            limits=[0.0019985, 0.0020015],
            offset=0.002,
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
        title=r"Mono X'Z'")
    plots.append(plot05)

    plot07 = xrtplot.XYCPlot(
        beam=r"screen04beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x'",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z'",
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
        title=r"Exit X'Z'")
    plots.append(plot07)

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
        title=r"Mirror XZ")
    plots.append(plot04)

    plot06 = xrtplot.XYCPlot(
        beam=r"screen03beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-1, 1],
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
        title=r"Mono XZ")
    plots.append(plot06)

    plot08 = xrtplot.XYCPlot(
        beam=r"screen04beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1,
            fwhmFormatStr=r"%.1e"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-1, 1],
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
        title=r"Exit XZ")
    plots.append(plot08)
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
