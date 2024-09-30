from typing import List
from xrt.backends.raycing import BeamLine
import xrt.runner as xrtrun
import xrt.backends.raycing.run as rrun


class Ptycho(BeamLine):
    pass


def run_process(bl: Ptycho):
    outDict = dict()
    bl.prepare_flow()
    return outDict


rrun.run_process = run_process


plots = []


def no_scan(bl: Ptycho, plots_: List):
    yield


if __name__ == "__main__":
    beamline = Ptycho()
    scan = no_scan
    show = True
    repeats = 1

    if show:
        beamline.glow(
            scale=[1e1, 1e4, 1e4],
            centerAt=r"Lens_03_Exit",
            generator=scan,
            generatorArgs=[beamline, plots],
            startFrom=1,
        )
    else:
        xrtrun.run_ray_tracing(
            beamLine=beamline,
            plots=plots,
            repeats=repeats,
            backend=r"raycing",
            generator=scan,
            generatorArgs=[beamline, plots],
        )
