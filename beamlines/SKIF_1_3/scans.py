from typing import List
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

from SKIF_1_3 import SKIF13 


# ################################### PLOTS ###################################


plots = [
    xrtplot.XYCPlot(
        beam='WgMonitorLocal',
        title='Wiggler Monitor',
        xaxis=xrtplot.XYCAxis(r'$x$', 'mm', limits=(0., 0.)),
        yaxis=xrtplot.XYCAxis(r'$z$', 'mm', limits=(0., 0.)),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='FEMonitorLocal',
        title='Front-End Monitor',
        xaxis=xrtplot.XYCAxis(r'$x$', 'mm', limits=(0., 0.)),
        yaxis=xrtplot.XYCAxis(r'$z$', 'mm', limits=(0., 0.)),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='LensEntranceMonitorLocal',
        title='Lens Entrance Monitor',
        xaxis=xrtplot.XYCAxis(r'$x$', 'mm', limits=(0., 0.)),
        yaxis=xrtplot.XYCAxis(r'$z$', 'mm', limits=(0., 0.)),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='LensExitMonitorLocal',
        title='Lens Exit Monitor',
        xaxis=xrtplot.XYCAxis(r'$x$', 'mm', limits=(0., 0.)),
        yaxis=xrtplot.XYCAxis(r'$z$', 'mm', limits=(0., 0.)),
        aspect='auto'),
    xrtplot.XYCPlot(
        beam='SampleMonitorLocal',
        title='Sample Monitor',
        xaxis=xrtplot.XYCAxis(r'$x$', 'mm', limits=(0., 0.)),
        yaxis=xrtplot.XYCAxis(r'$z$', 'mm', limits=(0., 0.)),
        aspect='auto'),
]


# ################################### SCANS ###################################


def none_scan(plts: List[xrtplot.XYCPlot], bl: SKIF13):
    # setting up axes limits
    for plt in plts:
        if plt.beam in ('WgMonitorLocal', 'FEMonitorLocal'):
            plt.xaxis.limits = 1.4 * bl.FrontEnd.opening[0], 1.4 * bl.FrontEnd.opening[1]
            plt.yaxis.limits = 1.4 * bl.FrontEnd.opening[2], 1.4 * bl.FrontEnd.opening[3]
        if plt.beam in ('LensEntranceMonitorLocal', 'LensExitMonitorLocal'):
            plt.xaxis.limits = 1.4 * bl.CrocLensStack[0].limPhysX[0], 1.4 * bl.CrocLensStack[0].limPhysX[1]
            plt.yaxis.limits = 1.4 * bl.CrocLensStack[0].limPhysY[0], 1.4 * bl.CrocLensStack[0].limPhysY[1]
            if plt.beam == 'LensExitMonitorLocal':
                plt.yaxis.limits = -.7 * bl.lens_pars['Aperture'], .7 * bl.lens_pars['Aperture']
        if plt.beam == 'SampleMonitorLocal':
            plt.xaxis.limits = 1.4 * bl.SampleSlit.opening[0], 1.4 * bl.SampleSlit.opening[1]
            plt.yaxis.limits = -.7 * bl.lens_pars['Aperture'], .7 * bl.lens_pars['Aperture']
    
    # setting up plot titles
    for plt in plts:
        if plt.beam == 'WgMonitorLocal':
            plt.title = ' '.join((plt.title, '[%.01f m]' % (bl.WigglerMonitor.center[1] * 1e-3)))
        elif plt.beam == 'FEMonitorLocal':
            plt.title = ' '.join((plt.title, '[%.01f m]' % (bl.FrontEndMonitor.center[1] * 1e-3)))
        elif plt.beam == 'LensEntranceMonitorLocal':
            plt.title = ' '.join((plt.title, '[%.01f m]' % (bl.LensEntranceMonitor.center[1] * 1e-3)))
        elif plt.beam == 'LensExitMonitorLocal':
            plt.title = ' '.join((plt.title, '[%.01f m]' % (bl.LensExitMonitor.center[1] * 1e-3)))
        elif plt.beam == 'SampleMonitorLocal':
            plt.title = ' '.join((plt.title, '[%.01f m]' % (bl.SampleMonitor.center[1] * 1e-3)))

    yield


# ################################### MAIN ####################################


if __name__ == '__main__':
    beamline = SKIF13()
    scan = none_scan
    show = False
    repeats = 10

    if show:
        beamline.glow(
            scale=[1e3, 1e3, 1e3],
            centerAt=2,
            generator=scan,
            generatorArgs=[plots, beamline],
            startFrom=1
        )
    else:
        xrtrun.run_ray_tracing(
            beamLine=beamline,
            plots=plots,
            repeats=repeats,
            backend=r"raycing",
            generator=scan,
            generatorArgs=[plots, beamline]
        )
