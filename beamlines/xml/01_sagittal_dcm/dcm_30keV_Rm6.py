# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-05-13"

Created with xrtQook




"""

import numpy as np
import sys
sys.path.append(r"/nix/store/mii96qf5vy744b67qmqgdars3iqjjpvv-python3.12-xrt-1.6.1/lib/python3.12/site-packages")
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


def build_beamline():
    beamLine = raycing.BeamLine(
        alignE=30000.0)

    beamLine.wiggler = rsources.Wiggler(
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
        eMin=29850,
        eMax=30150,
        eN=101,
        K=20.1685,
        period=48.0,
        n=18)

    beamLine.screenSource = rscreens.Screen(
        bl=beamLine,
        name=None)

    beamLine.screenFE = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 20000, 0])

    beamLine.dcmC1 = roes.BentLaue2D(
        bl=beamLine,
        name=None,
        center=[0, 33000, 0],
        pitch=2.25285,
        material=crystalSi01,
        alpha=0.6161012259539983,
        Rm=12360.0,
        Rs=-2060,
        targetOpenCL=[2, 0])

    beamLine.screenDCMC1 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 33020, 0])

    beamLine.dcmC2 = roes.BentLaue2D(
        bl=beamLine,
        name=None,
        center=[0, 33189.355, 25],
        pitch=2.12095,
        positionRoll=3.141592653589793,
        material=crystalSi01,
        alpha=0.6161012259539983,
        Rm=12360.0,
        Rs=-2060,
        targetOpenCL=[2, 0])

    beamLine.screenDCMC2 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 33200, 25])

    return beamLine


def run_process(beamLine):
    wiggler01beamGlobal01 = beamLine.wiggler.shine()

    screen04beamLocal01 = beamLine.screenSource.expose(
        beam=wiggler01beamGlobal01)

    screen01beamLocal01 = beamLine.screenFE.expose(
        beam=wiggler01beamGlobal01)

    bentLaue2D01beamGlobal01, bentLaue2D01beamLocal01 = beamLine.dcmC1.reflect(
        beam=wiggler01beamGlobal01)

    screen02beamLocal01 = beamLine.screenDCMC1.expose(
        beam=bentLaue2D01beamGlobal01)

    bentLaue2D02beamGlobal01, bentLaue2D02beamLocal01 = beamLine.dcmC2.reflect(
        beam=bentLaue2D01beamGlobal01)

    screen03beamLocal01 = beamLine.screenDCMC2.expose(
        beam=bentLaue2D02beamGlobal01)

    outDict = {
        'wiggler01beamGlobal01': wiggler01beamGlobal01,
        'screen04beamLocal01': screen04beamLocal01,
        'screen01beamLocal01': screen01beamLocal01,
        'bentLaue2D01beamGlobal01': bentLaue2D01beamGlobal01,
        'bentLaue2D01beamLocal01': bentLaue2D01beamLocal01,
        'screen02beamLocal01': screen02beamLocal01,
        'bentLaue2D02beamGlobal01': bentLaue2D02beamGlobal01,
        'bentLaue2D02beamLocal01': bentLaue2D02beamLocal01,
        'screen03beamLocal01': screen03beamLocal01}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    plot01 = xrtplot.XYCPlot(
        beam=r"screen04beamLocal01",
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
        aspect=r"auto",
        title=r"SOURCE XZ",
        persistentName=r"dcm_30keV_source_xz.npy")
    plots.append(plot01)

    plot06 = xrtplot.XYCPlot(
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
        aspect=r"auto",
        title=r"FRONTEND XZ",
        persistentName=r"dcm_30keV_frontend_xz.npy")
    plots.append(plot06)

    plot07 = xrtplot.XYCPlot(
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
        aspect=r"auto",
        title=r"C1 XZ",
        persistentName=r"dcm_30keV_c1_xz.npy")
    plots.append(plot07)

    plot10 = xrtplot.XYCPlot(
        beam=r"screen03beamLocal01",
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
        aspect=r"auto",
        title=r"C2 XZ",
        persistentName=r"dcm_30keV_c2_xz.npy")
    plots.append(plot10)

    plot02 = xrtplot.XYCPlot(
        beam=r"screen04beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"x'",
            unit=r"",
            bins=512,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        aspect=r"auto",
        title=r"SOURCE FX",
        persistentName=r"dcm_30keV_source_fx.npy")
    plots.append(plot02)

    plot05 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"x'",
            unit=r"",
            bins=512,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        aspect=r"auto",
        title=r"FRONTEND FX",
        persistentName=r"dcm_30keV_frontend_fx.npy")
    plots.append(plot05)

    plot08 = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"x'",
            unit=r"",
            bins=512,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        aspect=r"auto",
        title=r"C1 FX",
        persistentName=r"dcm_30keV_c1_fx.npy")
    plots.append(plot08)

    plot11 = xrtplot.XYCPlot(
        beam=r"screen03beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=512,
            ppb=1),
        yaxis=xrtplot.XYCAxis(
            label=r"x'",
            unit=r"",
            bins=512,
            ppb=1),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=512,
            ppb=1),
        aspect=r"auto",
        title=r"C2 FX",
        persistentName=r"dcm_30keV_c2_fx.npy")
    plots.append(plot11)

    plot03 = xrtplot.XYCPlot(
        beam=r"screen04beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"z",
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
        aspect=r"auto",
        title=r"SOURCE FZ",
        persistentName=r"dcm_30keV_source_fz.npy")
    plots.append(plot03)

    plot04 = xrtplot.XYCPlot(
        beam=r"screen01beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"z",
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
        aspect=r"auto",
        title=r"FRONTEND FZ",
        persistentName=r"dcm_30keV_frontend_fz.npy")
    plots.append(plot04)

    plot09 = xrtplot.XYCPlot(
        beam=r"screen02beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"z",
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
        aspect=r"auto",
        title=r"C1 FZ",
        persistentName=r"dcm_30keV_c1_fz.npy")
    plots.append(plot09)

    plot12 = xrtplot.XYCPlot(
        beam=r"screen03beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"z",
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
        aspect=r"auto",
        title=r"C2 FZ",
        persistentName=r"dcm_30keV_c2_fz.npy")
    plots.append(plot12)
    return plots


def main():
    beamLine = build_beamline()
    E0 = 0.5 * (beamLine.wiggler.eMin +
                beamLine.wiggler.eMax)
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=100,
        pickleEvery=1,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
