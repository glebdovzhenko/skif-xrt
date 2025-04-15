# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-04-02"

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

crystalSi01 = rmats.CrystalSi(
    t=0.5,
    geom=r"Laue reflected",
    name=None,
    useTT=True)


def build_beamline():
    beamLine = raycing.BeamLine(
        alignE=30000.0)

    beamLine.wiggler01 = rsources.Wiggler(
        bl=beamLine,
        center=[0, 0, 0],
        nrays=1000000,
        eE=3,
        eI=0.4,
        eEspread=0.00135,
        eEpsilonX=0.09586,
        eEpsilonZ=0.009586,
        betaX=15.66,
        betaZ=2.29,
        xPrimeMax=2,
        zPrimeMax=0.2,
        eMin=29750,
        eMax=30250,
        K=20.1685,
        period=48,
        n=18)

    beamLine.screen01 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 1000, 0])

    beamLine.bentLaue2D01 = roes.BentLaue2D(
        bl=beamLine,
        name=r"DCM C1",
        center=[0, 33000, 0],
        pitch=r"auto",
        material=crystalSi01,
        alpha=0.6161012259539983,
        Rm=-15000.0,
        Rs=5000.0,
        targetOpenCL=[2, 0],
        precisionOpenCL=r"float32")

    beamLine.screen02 = rscreens.Screen(
        bl=beamLine,
        name=r"Exit screen",
        center=[0, 56000, r"auto"])

    return beamLine


def run_process(beamLine):
    wiggler01beamGlobal01 = beamLine.wiggler01.shine()

    screen01beamLocal01 = beamLine.screen01.expose(
        beam=wiggler01beamGlobal01)

    bentLaue2D01beamGlobal01, bentLaue2D01beamLocal01 = beamLine.bentLaue2D01.reflect(
        beam=wiggler01beamGlobal01)

    screen02beamLocal01 = beamLine.screen02.expose(
        beam=bentLaue2D01beamGlobal01)

    outDict = {
        'wiggler01beamGlobal01': wiggler01beamGlobal01,
        'screen01beamLocal01': screen01beamLocal01,
        'bentLaue2D01beamGlobal01': bentLaue2D01beamGlobal01,
        'bentLaue2D01beamLocal01': bentLaue2D01beamLocal01,
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
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            bins=512,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        title=r"plot01")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            bins=512,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        title=r"plot02")
    plots.append(plot02)
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
