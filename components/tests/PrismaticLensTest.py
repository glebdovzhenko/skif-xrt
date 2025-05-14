import numpy as np
from matplotlib import pyplot as plt
from typing import List

import xrt.backends.raycing as raycing
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.sources as rsources
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from params.params_nstu_scw import (
    croc_crl_distance,
    optimal_croc_geometry,
    croc_geometry_BSU,
    monochromator_x_lim,
    monochromator_y_lim,
)
from components import PrismaticLens


# ################################ MATERIALS ##################################
mGlassyCarbon = rm.Material("C", rho=1.50, kind="lens")
mBeryllium = rm.Material("Be", rho=1.848, kind="lens")
lens_material = mBeryllium
crl_y_t = croc_geometry_BSU["Be"]["y_t"]
crl_L = croc_geometry_BSU["Be"]["L"]
crl_N = croc_geometry_BSU["Be"]["N"]


# ############################ SETUP PARAMETERS ###############################
en, d_en = 90e3, 1.0


# ################################## BEAMLINE #################################
class CrlTest(raycing.BeamLine):

    def __init__(self):
        raycing.BeamLine.__init__(self)
        self.name = "Prismatic Lens Test"

        self.GSource = rsources.GeometricSource(
            name="Geometric Source",
            bl=self,
            center=[0, 0, 0],
            distE="normal",
            energies=[en, d_en],
            distxprime="flat",
            distzprime="flat",
            dxprime=2.0e-3,
            dzprime=0.2e-3,
        )

        fdist = croc_crl_distance / 2.0
        crl_y_g = PrismaticLens.calc_y_g(lens_material, fdist, en, crl_y_t, crl_L)
        self.CrocLensStack = PrismaticLens.make_stack(
            L=crl_L,
            N=crl_N,
            d=crl_y_t,
            g_last=0.0,
            g_first=crl_y_g,
            bl=self,
            center=[0.0, croc_crl_distance, 0],
            material=lens_material,
            limPhysX=monochromator_x_lim,
            limPhysY=monochromator_y_lim,
        )

        self.SourceScreen = rscreens.Screen(
            bl=self,
            name=r"Lens Monitor",
            center=[0, 0, 0],
        )

        self.PreLensScreen = rscreens.Screen(
            bl=self,
            name=r"Lens Monitor",
            center=[0, croc_crl_distance - 10, 0],
        )

        self.PostLensScreen = rscreens.Screen(
            bl=self,
            name=r"Lens Monitor",
            center=[0, croc_crl_distance + crl_L + 10, 0],
        )

        self.ImageScreen = rscreens.Screen(
            bl=self,
            name=r"Lens Monitor",
            center=[0, 2.0 * croc_crl_distance, 0],
        )
        self.ProjectionScreen = rscreens.Screen(
            bl=self,
            name=r"Lens Monitor",
            center=[0, 2.0 * croc_crl_distance, 0],
        )


# ############################# BEAM TOPOLOGY #################################
def run_process(bl: CrlTest):

    beam_source = bl.sources[0].shine()
    beam_source_local = bl.SourceScreen.expose(beam=beam_source)
    beam_precrl_local = bl.PreLensScreen.expose(beam=beam_source)
    beam_faux_local = bl.ProjectionScreen.expose(beam=beam_source)

    outDict = {
        "BeamSourceGlobal": beam_source,
        "BeamSourceLocal": beam_source_local,
        "BeamPreCrlLocal": beam_precrl_local,
    }

    beamIn = beam_source
    # CRL
    for ilens, lens in enumerate(bl.CrocLensStack):
        lglobal, llocal1, llocal2 = lens.double_refract(beamIn, needLocal=True)
        strl = "_{0:02d}".format(ilens)
        outDict["BeamLensGlobal" + strl] = lglobal
        outDict["BeamLensLocal1" + strl] = llocal1
        outDict["BeamLensLocal2" + strl] = llocal2

        llocal2a = raycing.sources.Beam(copyFrom=llocal2)
        llocal2a.absorb_intensity(beamIn)
        outDict["BeamLensLocal2a" + strl] = llocal2a
        beamIn = lglobal

    beam_crl_exit = bl.PostLensScreen.expose(beam=beamIn)
    beam_image = bl.ImageScreen.expose(beam=beamIn)

    outDict["BeamPostCrlLocal"] = beam_crl_exit
    outDict["BeamImageLocal"] = beam_image
    outDict["BeamProjectionLocal"] = beam_faux_local

    bl.prepare_flow()
    return outDict


rrun.run_process = run_process


# ############################## ONEPT SCAN ###################################
def onept_plots(*args, **kwargs):
    extra_kwargs = {
        r"fwhmFormatStr": "%.2e",
        "bins": 256,
        "ppb": 1,
    }
    x_kwds = {r"label": r"$x$", r"unit": r"mm", r"data": raycing.get_x, **extra_kwargs}
    y_kwds = {r"label": r"$y$", r"unit": r"mm", r"data": raycing.get_y, **extra_kwargs}
    z_kwds = {
        r"label": r"$z$",
        r"unit": r"mm",
        r"data": raycing.get_z,
    }
    xpr_kwds = {
        r"label": r"$x^{\prime}$",
        r"unit": r"",
        r"data": raycing.get_xprime,
        **extra_kwargs,
    }
    zpr_kwds = {
        r"label": r"$z^{\prime}$",
        r"unit": r"",
        r"data": raycing.get_zprime,
        **extra_kwargs,
    }
    result = []

    for beam, t1 in zip(
        (
            "BeamSourceLocal",
            "BeamPreCrlLocal",
            "BeamPostCrlLocal",
            "BeamImageLocal",
        ),
        ("S", "CRL entrance", "CRL exit", "Image"),
    ):
        params = zip(
            ("XZ", "XXpr", "ZZpr"),
            (x_kwds, x_kwds, z_kwds),
            (z_kwds, xpr_kwds, zpr_kwds),
        )

        for t2, xkw, ykw in params:
            result.append(
                xrtplot.XYCPlot(
                    beam=beam,
                    title="-".join((t1, t2)),
                    xaxis=xrtplot.XYCAxis(**xkw),
                    yaxis=xrtplot.XYCAxis(**ykw),
                    aspect="auto",
                )
            )
    return result


def onept(bl: CrlTest, plts: List):
    for plot in plts:
        if "CRL exit-XZ" in plot.title:
            plot.yaxis.limits = [-1, 1]
        if "Image-XZ" in plot.title or "S-XZ" in plot.title:
            plot.yaxis.limits = [-0.1, 0.1]
    yield


# ################################ y_g SCAN ###################################
def y_g_scan_plots(*args, **kwargs):
    x_kwds = {r"label": r"$x$", r"unit": r"mm", r"data": raycing.get_x}
    y_kwds = {r"label": r"$y$", r"unit": r"mm", r"data": raycing.get_y}
    z_kwds = {
        r"label": r"$z$",
        r"unit": r"mm",
        r"data": raycing.get_z,
        r"fwhmFormatStr": "%.2e",
        "bins": 256,
        "ppb": 1,
    }
    xpr_kwds = {r"label": r"$x^{\prime}$", r"unit": r"", r"data": raycing.get_xprime}
    zpr_kwds = {r"label": r"$z^{\prime}$", r"unit": r"", r"data": raycing.get_zprime}
    result = []

    for beam, t1 in zip(
        (
            "BeamSourceLocal",
            "BeamImageLocal",
        ),
        ("S", "Image"),
    ):
        params = zip(
            ("XZ", "XXpr", "ZZpr"),
            (x_kwds, x_kwds, z_kwds),
            (z_kwds, xpr_kwds, zpr_kwds),
        )

        for t2, xkw, ykw in params:
            result.append(
                xrtplot.XYCPlot(
                    beam=beam,
                    title="-".join((t1, t2)),
                    xaxis=xrtplot.XYCAxis(**xkw),
                    yaxis=xrtplot.XYCAxis(**ykw),
                    aspect="auto",
                )
            )
    return result


def y_g_scan(bl: CrlTest, plts: List):
    fdist = croc_crl_distance / 2.0
    crl_y_g = PrismaticLens.calc_y_g(lens_material, fdist, en, crl_y_t, crl_L)

    result = {"y_g": [], "dz": []}

    # optimal results
    # 30 keV y_g = 0.937
    # 50 keV y_g = 0.337
    # 70 keV y_g = 0.170
    # 90 keV y_g = 0.105

    # for y_g in np.linspace(0.095, 0.11, 11):
    for y_g in crl_y_g * np.linspace(0.8, 1.2, 11):
        bl.CrocLensStack = PrismaticLens.make_stack(
            L=crl_L,
            N=int(crl_L),
            d=crl_y_t,
            g_last=0.0,
            g_first=y_g,
            bl=bl,
            center=[0.0, croc_crl_distance, 0],
            material=lens_material,
            limPhysX=monochromator_x_lim,
            limPhysY=monochromator_y_lim,
        )
        for plot in plts:
            if "Image-XZ" in plot.title or "S-XZ" in plot.title:
                plot.yaxis.limits = [-0.1, 0.1]
        yield

        result["y_g"].append(y_g)
        print(f"y_g = {y_g}")
        for plot in plts:
            if "Image-XZ" in plot.title:
                print(f"Image {plot.dy}")
                result["dz"].append(plot.dy)
            if "S-XZ" in plot.title:
                print(f"Source: {plot.dy}")

    fig = plt.figure()
    plt.plot(result["y_g"], result["dz"])
    fig.show()


# ################################ gain SCAN ##################################
def gain_scan_plots(*args, **kwargs):
    x_kwds = {r"label": r"$x$", r"unit": r"mm", r"data": raycing.get_x}
    y_kwds = {r"label": r"$y$", r"unit": r"mm", r"data": raycing.get_y}
    z_kwds = {
        r"label": r"$z$",
        r"unit": r"mm",
        r"data": raycing.get_z,
        r"fwhmFormatStr": "%.2e",
        "bins": 256,
        "ppb": 1,
    }
    xpr_kwds = {r"label": r"$x^{\prime}$", r"unit": r"", r"data": raycing.get_xprime}
    zpr_kwds = {r"label": r"$z^{\prime}$", r"unit": r"", r"data": raycing.get_zprime}
    result = []

    for beam, t1 in zip(
        ("BeamSourceLocal", "BeamImageLocal", "BeamProjectionLocal"),
        ("S", "Image", "Projection"),
    ):
        params = zip(
            ("XZ",),
            (x_kwds,),
            (z_kwds,),
        )

        for t2, xkw, ykw in params:
            result.append(
                xrtplot.XYCPlot(
                    beam=beam,
                    title="-".join((t1, t2)),
                    xaxis=xrtplot.XYCAxis(**xkw),
                    yaxis=xrtplot.XYCAxis(**ykw),
                    aspect="auto",
                )
            )
    return result


def gain_scan(bl: CrlTest, plts: List):
    fdist = croc_crl_distance / 2.0

    ens = (30e3, 50e3, 70e3, 90e3)
    d_ens = (1.0, 1.0, 1.0, 1.0)
    y_gs = (0.937, 0.337, 0.170, 0.105)

    for en, d_en, y_g in zip(ens, d_ens, y_gs):
        bl.GSource.energies = (en, d_en)
        bl.CrocLensStack = PrismaticLens.make_stack(
            L=crl_L,
            N=int(crl_L),
            d=crl_y_t,
            g_last=0.0,
            g_first=y_g,
            bl=bl,
            center=[0.0, croc_crl_distance, 0],
            material=lens_material,
            limPhysX=monochromator_x_lim,
            limPhysY=monochromator_y_lim,
        )
        for plot in plts:
            plot.xaxis.limits = None
            plot.yaxis.limits = None
            plot.caxis.limits = None

            if "Image-XZ" in plot.title or "S-XZ" in plot.title:
                plot.yaxis.limits = [-0.1, 0.1]
        yield

        gain = 1.0
        for plot in plts:
            if "Image-XZ" in plot.title:
                gain *= plot.intensity
                gain /= plot.dy
            if "Projection-XZ" in plot.title:
                gain /= plot.intensity
                gain *= plot.dy
        print(f"E = {en}, gain = {gain}")


# ################################## RUN ######################################
if __name__ == "__main__":
    beamline = CrlTest()
    scan = onept
    plot_gen = onept_plots
    show = False
    repeats = 4

    if show:
        beamline.glow(
            scale=[1e3, 1e3, 1e3],
            generator=scan,
            generatorArgs=[beamline, []],
            startFrom=1,
        )
    else:
        plots = plot_gen(beamline)
        xrtrun.run_ray_tracing(
            beamLine=beamline,
            plots=plots,
            repeats=repeats,
            backend=r"raycing",
            generator=scan,
            generatorArgs=[beamline, plots],
        )
