# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2025-04-05"

Created with xrtQook




Powder Diffraction
------------------

Simulation of the real powder diffraction experiment on PETRA-III
High Resolution Powder Diffraction Beamline P02.1.
Uses Undulator source and double Laue Plate monochromator.
Cerium Dioxide powder as the sample.

.. imagezoom:: _images/rings_on_detector.png
   :scale: 60%

.. warning::
   Heavy computational load. Requires OpenCL.




"""

import numpy as np
import sys
sys.path.append(r"/nix/store/gcn7513l4lmpwywg97af187cm2b3zfm1-python3.12-xrt-1.6.1/lib/python3.12/site-packages")
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

LaueXtal = rmats.CrystalHarmonics(
    Nmax=1,
    name=r"LaueMono",
    a=5.41949,
    t=0.2,
    geom=r"Laue reflected",
    table=r"Chantler")

PowderSample = rmats.Powder(
    chi=[0, 6.283185307179586],
    name=None,
    hkl=[7, 7, 7],
    a=5.256,
    atoms=[58, 58, 58, 58, 8, 8, 8, 8, 8, 8, 8, 8],
    atomsXYZ=[[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25], [0.75, 0.75, 0.75], [0.75, 0.25, 0.25], [0.25, 0.75, 0.25], [0.25, 0.25, 0.75]],
    tK=297.15,
    t=1.0,
    table=r"Chantler")

crystalSi01 = rmats.CrystalSi(
    table=r"Chantler",
    name=None)


def build_beamline():
    P02_2 = raycing.BeamLine()

    P02_2.Undulator01 = rsources.Undulator(
        bl=P02_2,
        name=r"P02_U23",
        center=[0, 0, 0],
        nrays=10000000,
        eE=6.08,
        eI=0.2,
        eEspread=0.001,
        betaX=20.01,
        betaZ=2.36,
        xPrimeMax=0.0020689655172413794,
        zPrimeMax=0.0020689655172413794,
        targetE=[60000, 11],
        eMin=59940,
        eMax=60060,
        K=10.0,
        Ky=0.0,
        period=23,
        n=87,
        gp=1e-06,
        targetOpenCL=[0, 0],
        precisionOpenCL=r"float32")

    P02_2.SLIT_FE = rapts.RectangularAperture(
        bl=P02_2,
        name=r"FrontEndSlit",
        center=[r"auto", 29000, r"auto"],
        opening=[-1.0, 1.0, -1.0, 1.0])

    P02_2.FSM_Source = rscreens.Screen(
        bl=P02_2,
        name=r"FSM_Source",
        center=[r"auto", 29001, r"auto"])

    P02_2.LP1 = roes.LauePlate(
        bl=P02_2,
        name=r"LauePlate1",
        center=[r"auto", 36000, r"auto"],
        pitch=r"auto",
        positionRoll=1.5707963267948966,
        material=LaueXtal,
        targetOpenCL=[0, 0],
        precisionOpenCL=r"float32")

    P02_2.LP2 = roes.LauePlate(
        bl=P02_2,
        name=r"LauePlate2",
        center=[r"auto", 44000, r"auto"],
        pitch=r"auto",
        positionRoll=-1.5707963267948966,
        material=LaueXtal,
        targetOpenCL=[0, 0],
        precisionOpenCL=r"float32")

    P02_2.FSM_DCM = rscreens.Screen(
        bl=P02_2,
        name=r"DCM_Screen",
        center=[r"auto", 44500, r"auto"])

    P02_2.Slit_EH = rapts.RectangularAperture(
        bl=P02_2,
        name=r"ExperimentalHutchSlit",
        center=[r"auto", 62000, r"auto"],
        opening=[-1.0, 1.0, -1.0, 1.0])

    P02_2.PowderSample = roes.LauePlate(
        bl=P02_2,
        name=r"CeO2 Powder",
        center=[r"auto", 65000, r"auto"],
        pitch=1.5707963267948966,
        material=PowderSample,
        targetOpenCL=[0, 0],
        precisionOpenCL=r"float32")

    P02_2.FSM_Sample = rscreens.Screen(
        bl=P02_2,
        name=None,
        center=[P02_2.PowderSample.center[0], 65100, P02_2.PowderSample.center[2]])

    P02_2.RoundBeamStop01 = rapts.RoundBeamStop(
        bl=P02_2,
        name=r"BeamStop",
        center=[P02_2.PowderSample.center[0], 65499, P02_2.PowderSample.center[2]])

    P02_2.FSM_Detector = rscreens.Screen(
        bl=P02_2,
        name=r"Detector",
        center=[P02_2.PowderSample.center[0], 65500, P02_2.PowderSample.center[2]])

    return P02_2


def run_process(P02_2):
    Undulator01beamGlobal01 = P02_2.Undulator01.shine(
        withAmplitudes=False)

    SLIT_FEbeamLocal01 = P02_2.SLIT_FE.propagate(
        beam=Undulator01beamGlobal01)

    FSM_SourcebeamLocal01 = P02_2.FSM_Source.expose(
        beam=Undulator01beamGlobal01)

    LP1beamGlobal01, LP1beamLocal01 = P02_2.LP1.reflect(
        beam=Undulator01beamGlobal01)

    LP2beamGlobal01, LP2beamLocal01 = P02_2.LP2.reflect(
        beam=LP1beamGlobal01)

    FSM_DCMbeamLocal01 = P02_2.FSM_DCM.expose(
        beam=LP2beamGlobal01)

    Slit_EHbeamLocal01 = P02_2.Slit_EH.propagate(
        beam=LP2beamGlobal01)

    PowderSamplebeamGlobal01, PowderSamplebeamLocal01 = P02_2.PowderSample.reflect(
        beam=LP2beamGlobal01)

    FSM_SamplebeamLocal01 = P02_2.FSM_Sample.expose(
        beam=PowderSamplebeamGlobal01)

    RoundBeamStop01beamLocal01 = P02_2.RoundBeamStop01.propagate(
        beam=PowderSamplebeamGlobal01)

    FSM_DetectorbeamLocal01 = P02_2.FSM_Detector.expose(
        beam=PowderSamplebeamGlobal01)

    outDict = {
        'Undulator01beamGlobal01': Undulator01beamGlobal01,
        'SLIT_FEbeamLocal01': SLIT_FEbeamLocal01,
        'FSM_SourcebeamLocal01': FSM_SourcebeamLocal01,
        'LP1beamGlobal01': LP1beamGlobal01,
        'LP1beamLocal01': LP1beamLocal01,
        'LP2beamGlobal01': LP2beamGlobal01,
        'LP2beamLocal01': LP2beamLocal01,
        'FSM_DCMbeamLocal01': FSM_DCMbeamLocal01,
        'Slit_EHbeamLocal01': Slit_EHbeamLocal01,
        'PowderSamplebeamGlobal01': PowderSamplebeamGlobal01,
        'PowderSamplebeamLocal01': PowderSamplebeamLocal01,
        'FSM_SamplebeamLocal01': FSM_SamplebeamLocal01,
        'RoundBeamStop01beamLocal01': RoundBeamStop01beamLocal01,
        'FSM_DetectorbeamLocal01': FSM_DetectorbeamLocal01}
    return outDict


rrun.run_process = run_process



def define_plots():
    plots = []

    Plot01 = xrtplot.XYCPlot(
        beam=r"FSM_SourcebeamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        title=r"01 - Undulator Beam at 29m",
        fluxFormatStr=r"%g",
        saveName=r"01 - Undulator Beam at 29m.png")
    plots.append(Plot01)

    Plot02 = xrtplot.XYCPlot(
        beam=r"FSM_DCMbeamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        title=r"02 - Monocromatized Beam",
        fluxFormatStr=r"%g",
        saveName=r"02 - Monocromatized Beam.png")
    plots.append(Plot02)

    Plot03 = xrtplot.XYCPlot(
        beam=r"FSM_DetectorbeamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-100, 100],
            bins=1000,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        yaxis=xrtplot.XYCAxis(
            label=r"z",
            limits=[-100, 100],
            bins=1000,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        title=r"03 - Detector",
        fluxFormatStr=r"%g",
        saveName=r"03 - Detector.png",
        persistentName=r"03 - Detector.npy")
    plots.append(Plot03)

    Plot04 = xrtplot.XYCPlot(
        beam=r"LP1beamLocal01",
        xaxis=xrtplot.XYCAxis(
            label=r"x",
            limits=[-1, 1],
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        yaxis=xrtplot.XYCAxis(
            label=r"y",
            limits=[-1, 1],
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        caxis=xrtplot.XYCAxis(
            label=r"energy",
            unit=r"eV",
            bins=256,
            ppb=1,
            fwhmFormatStr=r"%.2f"),
        title=r"04 -Laue Plate 1 Footprint",
        fluxFormatStr=r"%g",
        saveName=r"04 -Laue Plate 1 Footprint.png")
    plots.append(Plot04)
    return plots


def main():
    P02_2 = build_beamline()
    E0 = 0.5 * (P02_2.Undulator01.eMin +
                P02_2.Undulator01.eMax)
    P02_2.alignE=E0
    plots = define_plots()
    xrtrun.run_ray_tracing(
        plots=plots,
        repeats=1000,
        pickleEvery=10,
        backend=r"raycing",
        beamLine=P02_2)


if __name__ == '__main__':
    main()
