# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-05-13"

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

cvDDiamond01 = rmatsco.CVDDiamond()

siliconCarbide01 = rmatsco.SiliconCarbide(
    kind=r"lens")


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.wiggler01 = rsources.Wiggler(
        bl=beamLine,
        center=[0, 0, 0],
        nrays=1000000,
        eE=3,
        eI=0.4,
        eEspread=0.00135,
        eEpsilonX=0.075,
        eEpsilonZ=0.0075,
        betaX=15.66,
        betaZ=2.29,
        xPrimeMax=1,
        zPrimeMax=0.1,
        eMin=100.0,
        eMax=150000.0,
        eN=5001,
        K=20.1685,
        period=48.0,
        n=18)

    beamLine.fDia01 = roes.Plate(
        bl=beamLine,
        name=None,
        center=[0, 17959, 0],
        pitch=1.5707963267948966,
        material=cvDDiamond01,
        limPhysX=[-22.5, 22.5],
        limPhysY=[-5.0, 5.0],
        t=0.5)

    beamLine.fDia02 = roes.Plate(
        bl=beamLine,
        name=None,
        center=[0, 18071, 0],
        pitch=1.5707963267948966,
        material=cvDDiamond01,
        limPhysX=[-22.5, 22.5],
        limPhysY=[-5.0, 5.0],
        t=0.5)

    beamLine.fDia03 = roes.Plate(
        bl=beamLine,
        name=None,
        center=[0, 18183, 0],
        pitch=1.5707963267948966,
        material=cvDDiamond01,
        limPhysX=[-22.5, 22.5],
        limPhysY=[-5.0, 5.0],
        t=0.5)

    beamLine.fDia04 = roes.Plate(
        bl=beamLine,
        name=None,
        center=[0, 18779, 0],
        pitch=1.5707963267948966,
        material=cvDDiamond01,
        limPhysX=[-22.5, 22.5],
        limPhysY=[-5.0, 5.0],
        t=0.5)

    beamLine.fDia05 = roes.Plate(
        bl=beamLine,
        name=None,
        center=[0, 19380, 0],
        pitch=1.5707963267948966,
        material=cvDDiamond01,
        limPhysX=[-22.5, 22.5],
        limPhysY=[-5.0, 5.0],
        t=0.5)

    beamLine.fSiC01 = roes.Plate(
        bl=beamLine,
        name=None,
        center=[0, 22184, 0],
        pitch=1.5707963267948966,
        material=siliconCarbide01,
        limPhysX=[-30, 30],
        limPhysY=[-7.5, 7.5],
        t=0.35)

    beamLine.fSiC02 = roes.Plate(
        bl=beamLine,
        name=None,
        center=[0, 22296, 0],
        pitch=1.5707963267948966,
        material=siliconCarbide01,
        limPhysX=[-30, 30],
        limPhysY=[-7.5, 7.5],
        t=0.35)

    beamLine.fSiC03 = roes.Plate(
        bl=beamLine,
        name=None,
        center=[0, 22408, 0],
        pitch=1.5707963267948966,
        material=siliconCarbide01,
        limPhysX=[-30, 30],
        limPhysY=[-7.5, 7.5],
        t=0.35)

    beamLine.fSiC04 = roes.Plate(
        bl=beamLine,
        name=None,
        center=[0, 22520, 0],
        pitch=1.5707963267948966,
        material=siliconCarbide01,
        limPhysX=[-30, 30],
        limPhysY=[-7.5, 7.5],
        t=0.35)

    beamLine.fSiC05 = roes.Plate(
        bl=beamLine,
        name=None,
        center=[0, 25135, 0],
        pitch=1.5707963267948966,
        material=siliconCarbide01,
        limPhysX=[-30, 30],
        limPhysY=[-7.5, 7.5],
        t=0.35)

    beamLine.sOpticHutchEntrance01 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 28000, 0])

    return beamLine


def run_process(beamLine):
    wiggler01beamGlobal01 = beamLine.wiggler01.shine()

    plate01beamGlobal01, plate01beamLocal101, plate01beamLocal201 = beamLine.fDia01.double_refract(
        beam=wiggler01beamGlobal01,
        returnLocalAbsorbed=0)

    plate02beamGlobal01, plate02beamLocal101, plate02beamLocal201 = beamLine.fDia02.double_refract(
        beam=plate01beamGlobal01,
        returnLocalAbsorbed=0)

    plate03beamGlobal01, plate03beamLocal101, plate03beamLocal201 = beamLine.fDia03.double_refract(
        beam=plate02beamGlobal01,
        returnLocalAbsorbed=0)

    plate04beamGlobal01, plate04beamLocal101, plate04beamLocal201 = beamLine.fDia04.double_refract(
        beam=plate03beamGlobal01,
        returnLocalAbsorbed=0)

    plate05beamGlobal01, plate05beamLocal101, plate05beamLocal201 = beamLine.fDia05.double_refract(
        beam=plate04beamGlobal01,
        returnLocalAbsorbed=0)

    plate01beamGlobal02, plate01beamLocal102, plate01beamLocal202 = beamLine.fSiC01.double_refract(
        beam=plate05beamGlobal01,
        returnLocalAbsorbed=0)

    plate02beamGlobal02, plate02beamLocal102, plate02beamLocal202 = beamLine.fSiC02.double_refract(
        beam=plate01beamGlobal02,
        returnLocalAbsorbed=0)

    plate03beamGlobal02, plate03beamLocal102, plate03beamLocal202 = beamLine.fSiC03.double_refract(
        beam=plate02beamGlobal02,
        returnLocalAbsorbed=0)

    plate04beamGlobal02, plate04beamLocal102, plate04beamLocal202 = beamLine.fSiC04.double_refract(
        beam=plate03beamGlobal02,
        returnLocalAbsorbed=0)

    plate05beamGlobal02, plate05beamLocal102, plate05beamLocal202 = beamLine.fSiC05.double_refract(
        beam=plate04beamGlobal02,
        returnLocalAbsorbed=0)

    screen01beamLocal01 = beamLine.sOpticHutchEntrance01.expose(
        beam=plate05beamGlobal02)

    outDict = {
        'wiggler01beamGlobal01': wiggler01beamGlobal01,
        'plate01beamGlobal01': plate01beamGlobal01,
        'plate01beamLocal101': plate01beamLocal101,
        'plate01beamLocal201': plate01beamLocal201,
        'plate02beamGlobal01': plate02beamGlobal01,
        'plate02beamLocal101': plate02beamLocal101,
        'plate02beamLocal201': plate02beamLocal201,
        'plate03beamGlobal01': plate03beamGlobal01,
        'plate03beamLocal101': plate03beamLocal101,
        'plate03beamLocal201': plate03beamLocal201,
        'plate04beamGlobal01': plate04beamGlobal01,
        'plate04beamLocal101': plate04beamLocal101,
        'plate04beamLocal201': plate04beamLocal201,
        'plate05beamGlobal01': plate05beamGlobal01,
        'plate05beamLocal101': plate05beamLocal101,
        'plate05beamLocal201': plate05beamLocal201,
        'plate01beamGlobal02': plate01beamGlobal02,
        'plate01beamLocal102': plate01beamLocal102,
        'plate01beamLocal202': plate01beamLocal202,
        'plate02beamGlobal02': plate02beamGlobal02,
        'plate02beamLocal102': plate02beamLocal102,
        'plate02beamLocal202': plate02beamLocal202,
        'plate03beamGlobal02': plate03beamGlobal02,
        'plate03beamLocal102': plate03beamLocal102,
        'plate03beamLocal202': plate03beamLocal202,
        'plate04beamGlobal02': plate04beamGlobal02,
        'plate04beamLocal102': plate04beamLocal102,
        'plate04beamLocal202': plate04beamLocal202,
        'plate05beamGlobal02': plate05beamGlobal02,
        'plate05beamLocal102': plate05beamLocal102,
        'plate05beamLocal202': plate05beamLocal202,
        'screen01beamLocal01': screen01beamLocal01}
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
        title=r"Optic Hutch Entrance P")
    plots.append(plot01)

    plot02 = xrtplot.XYCPlot(
        beam=r"plate01beamLocal201",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Dia01 Pabs",
        fluxKind=r"power")
    plots.append(plot02)

    plot03 = xrtplot.XYCPlot(
        beam=r"plate02beamLocal201",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Dia02 Pabs",
        fluxKind=r"power")
    plots.append(plot03)

    plot04 = xrtplot.XYCPlot(
        beam=r"plate03beamLocal201",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Dia03 Pabs",
        fluxKind=r"power")
    plots.append(plot04)

    plot05 = xrtplot.XYCPlot(
        beam=r"plate04beamLocal201",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Dia04 Pabs",
        fluxKind=r"power")
    plots.append(plot05)

    plot06 = xrtplot.XYCPlot(
        beam=r"plate05beamLocal201",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"Dia05 Pabs",
        fluxKind=r"power")
    plots.append(plot06)

    plot07 = xrtplot.XYCPlot(
        beam=r"plate01beamLocal202",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"SiC01 Pabs",
        fluxKind=r"power")
    plots.append(plot07)

    plot08 = xrtplot.XYCPlot(
        beam=r"plate02beamLocal202",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"SiC02 Pabs",
        fluxKind=r"power")
    plots.append(plot08)

    plot09 = xrtplot.XYCPlot(
        beam=r"plate03beamLocal202",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"SiC03 Pabs",
        fluxKind=r"power")
    plots.append(plot09)

    plot10 = xrtplot.XYCPlot(
        beam=r"plate04beamLocal202",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"SiC04 Pabs",
        fluxKind=r"power")
    plots.append(plot10)

    plot11 = xrtplot.XYCPlot(
        beam=r"plate05beamLocal202",
        xaxis=xrtplot.XYCAxis(
            label=r"x"),
        yaxis=xrtplot.XYCAxis(
            label=r"y"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        title=r"SiC05 Pabs",
        fluxKind=r"power")
    plots.append(plot11)
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
