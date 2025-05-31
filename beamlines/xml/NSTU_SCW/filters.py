# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-05-16"

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

cvDDiamond01 = rmatsco.CVDDiamond()

siliconCarbide01 = rmatsco.SiliconCarbide(
    kind=r"lens")


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.wiggler01 = rsources.Wiggler(
        bl=beamLine,
        center=[0, 0, 0],
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

    beamLine.sOpticHutchEntrance01 = rscreens.Screen(
        bl=beamLine,
        name=None,
        center=[0, 28000, 0])

    return beamLine


def run_process(beamLine):
    wiggler01beamGlobal01 = beamLine.wiggler01.shine()

    screen01beamLocal01 = beamLine.sOpticHutchEntrance01.expose(
        beam=wiggler01beamGlobal01)

    outDict = {
        'wiggler01beamGlobal01': wiggler01beamGlobal01,
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
            label=r"z",
            fwhmFormatStr=r"%.1e"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV"),
        aspect=r"auto",
        title=r"Optic Hutch Entrance F")
    plots.append(plot01)
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
