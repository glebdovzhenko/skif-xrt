import xrt.plotter as xrtplot
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize
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


def get_line_kb(data: xrtplot.SaveResults, show=False):
    """
    :param data: assuming that y axis is z', x axis is z, c axis is energy
    :return:
    """

    xvals = .5 * (data.xbinEdges[1:] + data.xbinEdges[:-1])
    yvals = .5 * (data.ybinEdges[1:] + data.ybinEdges[:-1])
    xvals_, yvals_ = np.meshgrid(xvals, yvals)

    def f(x):
        return np.sum(data.total2D * (yvals_ - xvals_ * np.tan(x[0]) - x[1]) ** 2)

    min_res = minimize(f, np.array([.1, .1]), bounds=[(-np.pi/2, np.pi/2), (-1., 1.)])

    if show:
        _, ax = plt.subplots()
        plt.imshow(data.total2D, aspect='auto', origin='lower',
                   extent=[data.xbinEdges.min(), data.xbinEdges.max(),
                           data.ybinEdges.min(), data.ybinEdges.max()])
        plt.plot(xvals, xvals * np.tan(min_res.x[0]) + min_res.x[1])

        plt.xlim(data.xbinEdges.min(), data.xbinEdges.max())
        plt.ylim(data.ybinEdges.min(), data.ybinEdges.max())
        ax.text(.5 * (data.xbinEdges.min() + data.xbinEdges.max()), data.ybinEdges.max(), 
                'k=%e, \nb=%e' % (np.tan(min_res.x[0]), min_res.x[1]))
        plt.show()

    return np.tan(min_res.x[0]), min_res.x[1]


def get_minmax(data: xrtplot.SaveResults, axis: str ='x', fadeout: float=1e-3):
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

    t1d_xs = .5 * (be[1:] + be[:-1])
    t1d_dx = np.mean(be[1:] - be[:-1])
    t1d_xs = t1d_xs[t1d > fadeout * np.max(t1d)]
    
    return np.min(t1d_xs) - t1d_dx, np.max(t1d_xs) + t1d_dx


