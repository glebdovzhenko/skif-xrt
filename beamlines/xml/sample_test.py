# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-04-04"

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

powder01 = rmats.Powder(
    t=1,
    geom=r"Laue reflected")


def build_beamline():
    beamLine = raycing.BeamLine()

    beamLine.undulator01 = rsources.Undulator(
        bl=beamLine,
        center=[0, 0, 0],
        eE=3,
        eI=0.4,
        eEspread=0.00135,
        eEpsilonX=0.09586,
        eEpsilonZ=0.009586,
        betaX=15.66,
        betaZ=2.29,
        xPrimeMax=0.05,
        zPrimeMax=0.05,
        targetE=[15000, 7],
        eMin=14999,
        eMax=15001,
        period=15.6,
        n=128,
        targetOpenCL=[0, 0],
        precisionOpenCL=r"float32")

    beamLine.plate01 = roes.Plate(
        bl=beamLine,
        center=[0, 20000, 0],
        pitch=3.141592653589793,
        material=powder01,
        t=1,
        targetOpenCL=[2, 0],
        precisionOpenCL=r"float32")

    return beamLine


def run_process(beamLine):
    undulator01beamGlobal01 = beamLine.undulator01.shine()

    plate01beamGlobal01, plate01beamLocal101, plate01beamLocal201 = beamLine.plate01.double_refract(
        beam=undulator01beamGlobal01)

    outDict = {
        'undulator01beamGlobal01': undulator01beamGlobal01,
        'plate01beamGlobal01': plate01beamGlobal01,
        'plate01beamLocal101': plate01beamLocal101,
        'plate01beamLocal201': plate01beamLocal201}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []
    return plots


def main():
    beamLine = build_beamline()
    E0 = 0.5 * (beamLine.undulator01.eMin +
                beamLine.undulator01.eMax)
    beamLine.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        backend=r"raycing",
        beamLine=beamLine)


if __name__ == '__main__':
    main()
