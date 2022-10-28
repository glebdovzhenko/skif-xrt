import xrt.plotter as xrtplot
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import convolve
from uncertainties import ufloat


def reflectivity_box_fit(xd, yd):

    def f(x, *args):
        """
        FWHM = 2. * np.sqrt(2. * np.log(2)) * sigma
        :param x:
        :param args: [0] center, [1] heaviside width, [2] gaussian FWHM, [3] amplitude, [4] skew
        :return:
        """
        result = np.zeros(shape=x.shape)

        gs = args[2] / 2. * np.sqrt(2. * np.log(2))
        rw = (x - args[0]) > np.abs(.5 * args[1])
        lw = (x - args[0]) < -np.abs(.5 * args[1])
        cc = np.abs(x - args[0]) < np.abs(.5 * args[1])

        result[cc] = 1.

        result[rw] = np.exp(-((x[rw] - args[0] - .5 * np.abs(args[1])) / gs) ** 2 / 2.)
        result[lw] = np.exp(-((x[lw] - args[0] + .5 * np.abs(args[1])) / gs) ** 2 / 2.)

        result *= args[3]
        result *= args[4] * (x - args[0]) + 1.

        return result

    reflectivity_box_fit.f = f

    br = np.sum((.5 * yd[1:] + .5 * yd[:-1]) * (xd[1:] - xd[:-1])) / np.max(yd)
    p0 = [
        np.sum(xd * yd) / np.sum(yd),
        .5 * br,
        .5 * br,
        np.max(yd),
        0.
    ]

    try:
        return curve_fit(f, xd, yd, p0)
        # raise RuntimeError()
    except RuntimeError:
        return p0, None


def bell_fit(xd, yd, bckg=False, fn='gauss'):
    if fn == 'gauss':
        if not bckg:
            def f(x, *args):
                return np.exp(-((x - args[0]) / args[1])**2 / 2.) * args[2]
        else:
            def f(x, *args):
                return np.exp(-((x - args[0]) / args[1])**2 / 2.) * args[2] + args[3]
    elif fn == 'lorentz':
        if not bckg:
            def f(x, *args):
                return args[2] * args[1] ** 2 / ((x - args[0]) ** 2 + args[1] ** 2)
        else:
            def f(x, *args):
                return args[2] * args[1] ** 2 / ((x - args[0]) ** 2 + args[1] ** 2) + args[3]
    elif fn == 'heaviside':
        if not bckg:
            def f(x, *args):
                result = np.zeros(shape=x.shape)
                result[np.abs(x - args[0]) < args[1]] = args[2]
                result[x - args[0] > args[1]] = np.exp(
                    -((x[x - args[0] > args[1]] - args[0] - args[1]) / args[3])**2 / 2.) * args[2]
                result[x - args[0] < -args[1]] = np.exp(
                    -((x[x - args[0] < -args[1]] - args[0] + args[1]) / args[3]) ** 2 / 2.) * args[2]
                return result
        else:
            raise NotImplementedError()
    else:
        raise ValueError()

    bell_fit.f = f
    br = np.sum((.5 * yd[1:] + .5 * yd[:-1]) * (xd[1:] - xd[:-1])) / np.max(yd)
    p0 = [
        np.sum(xd * yd) / np.sum(yd),
        .5 * br,
        np.max(yd)
    ]

    if bckg:
        p0.append(0.)

    if fn == 'heaviside':
        p0.append(.25 * br,)
        p0[1] /= 2.

    try:
        return curve_fit(f, xd, yd, p0)
        # raise RuntimeError()
    except RuntimeError:
        return p0, None


def get_integral_breadth(data: xrtplot.SaveResults, axis: str = 'y'):
    if axis not in ('x', 'y', 'e'):
        raise ValueError()

    if axis == 'x':
        be = data.xbinEdges
        t1d = data.xtotal1D
    elif axis == 'y':
        be = data.ybinEdges
        t1d = data.ytotal1D
    elif axis == 'e':
        be = data.ebinEdges
        t1d = data.etotal1D
    else:
        return 0.

    bins = .5 * be[1:] + .5 * be[:-1]
    return np.sum((.5 * t1d[1:] + .5 * t1d[:-1]) * (bins[1:] - bins[:-1])) / np.max(t1d)


def get_line_kb(data: xrtplot.SaveResults, show=False, uncertainties=True):
    """
    :param data: assuming that y axis is z', x axis is z, c axis is energy
    :return:
    """

    x_centers = .5 * data.xbinEdges[1:] + .5 * data.xbinEdges[:-1]
    y_centers = .5 * data.ybinEdges[1:] + .5 * data.ybinEdges[:-1]

    ks = np.tan(np.linspace(.01 * np.pi, .49 * np.pi, 1000)) * \
        np.mean(data.ybinEdges[1:] - data.ybinEdges[:-1]) / \
        np.mean(data.xbinEdges[1:] - data.xbinEdges[:-1])
    b0 = np.mean(y_centers)

    x_centers, y_centers = np.meshgrid(x_centers, y_centers)

    if show:
        plt.imshow(data.total2D, aspect='auto', origin='lower',
                   extent=[data.xbinEdges.min(), data.xbinEdges.max(),
                           data.ybinEdges.min(), data.ybinEdges.max()])

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    def scan(k_vals):
        fs_vals = []
        for ii, k in enumerate(k_vals):
            offset = .5 * np.mean(data.ybinEdges[1:] - data.ybinEdges[:-1]) / np.cos(
                np.arctan(k / (np.mean(data.ybinEdges[1:] - data.ybinEdges[:-1]) / np.mean(
                    data.xbinEdges[1:] - data.xbinEdges[:-1]))))

            fs_vals.append(np.mean(data.total2D[
                (y_centers < k * x_centers + b0 + offset) & (y_centers > k * x_centers + b0 - offset)
            ]))

            if show:
                plt.plot(data.xbinEdges, k * data.xbinEdges + b0 + offset, color=colors[ii % len(colors)])
                plt.plot(data.xbinEdges, k * data.xbinEdges + b0 - offset, color=colors[ii % len(colors)])

        return fs_vals

    fs = scan(ks)
    fs = np.array(fs)
    k = ks[np.argmax(fs)]

    ks = np.linspace(ks.min(), k + 0.1 * (k + ks.max()), 1000)
    fs = np.array(scan(ks))

    popt, pcov = bell_fit(ks, fs)

    if show:
        plt.xlim(data.xbinEdges.min(), data.xbinEdges.max())
        plt.ylim(data.ybinEdges.min(), data.ybinEdges.max())
        plt.show()

    if uncertainties:
        return ufloat(popt[0], popt[1]), b0
    else:
        return popt[0], b0
