import numpy as np
import pickle
import os

from matplotlib import pyplot as plt

from utils.xrtutils import bell_fit, get_integral_breadth
from utils.various import datafiles


fig_width = 5.39  # inches
fig_aspect = 6 / 16
fig_aspect2 = 9 / 16


def flux_vs_e_chis():
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/e_scan_chi'

    alphas_plotted = [5, 15, 30, 40, 65, 70]
    alphas, es, flux = [], [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'C1C2' or metadata['axes'] != 'ZE':
            continue

        alphas.append(metadata['alpha'])
        es.append(metadata['energy'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)

    flux, alphas, es = np.array(flux), np.array(alphas), np.array(es)
    u_alphas, u_es = np.array(sorted(set(alphas))), np.array(sorted(set(es)))

    u_flux, u_z_breadth = np.nan + np.zeros(shape=(u_es.shape[0], u_alphas.shape[0])), \
                          np.nan + np.zeros(shape=(u_es.shape[0], u_alphas.shape[0]))

    for ii, alpha in enumerate(u_alphas):
        for jj, e in enumerate(u_es):
            u_flux[jj, ii] = flux[(alphas == alpha) & (es == e)].mean()

    fig = plt.figure()
    for ii, alp in enumerate(u_alphas):
        if alp in alphas_plotted:
            plt.semilogy(u_es, u_flux[:, ii], label='$\chi = %.01f ^{\circ}$' % alp)

    fig.set_size_inches(w=fig_width, h=fig_width*fig_aspect2)
    plt.title('Поток отражённый одним кристаллом')
    plt.xlabel(r'$h \nu$, кэВ')
    plt.ylabel('Поток, [ф/с]')
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=.4)

    if save:
        fig.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/flux_vs_e_chis.pgf')


def flux_vs_e_ts():
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/e_scan_t2'

    ts, es, flux = [], [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'C1C2' or metadata['axes'] != 'ZE':
            continue

        ts.append(metadata['thickness'])
        es.append(metadata['energy'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)

    flux, ts, es = np.array(flux), np.array(ts), np.array(es)
    u_ts, u_es = np.array(sorted(set(ts))), np.array(sorted(set(es)))

    u_flux, u_z_breadth = np.nan + np.zeros(shape=(u_es.shape[0], u_ts.shape[0])), \
                          np.nan + np.zeros(shape=(u_es.shape[0], u_ts.shape[0]))

    for ii, t in enumerate(u_ts):
        for jj, e in enumerate(u_es):
            u_flux[jj, ii] = flux[(ts == t) & (es == e)].mean()

    fig = plt.figure()
    for ii, alp in enumerate(u_ts):
        plt.semilogy(u_es, u_flux[:, ii], label='$t = %.01f$ мм' % alp)

    fig.set_size_inches(w=fig_width, h=fig_width*fig_aspect2)
    plt.title('Поток отражённый одним кристаллом')
    plt.xlabel(r'$h \nu$, кэВ')
    plt.ylabel('Поток, [ф/с]')
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=.4)

    if save:
        fig.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/flux_vs_e_ts2.pgf')


def flux_vs_r1r2():
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/r1r2_map'

    r1s, r2s, flux = [], [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'ES' or metadata['axes'] != 'XZ':
            continue

        r1s.append(metadata['r1'])
        r2s.append(metadata['r2'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)

    r1s, r2s, flux = np.array(r1s), np.array(r2s), np.array(flux)
    u_r1s, u_r2s = np.array(sorted(set(r1s))), np.array(sorted(set(r2s)))

    u_flux, u_z_breadth = np.nan + np.zeros(shape=(u_r1s.shape[0], u_r2s.shape[0])), \
                          np.nan + np.zeros(shape=(u_r1s.shape[0], u_r2s.shape[0]))

    for ii, r1 in enumerate(u_r1s):
        for jj, r2 in enumerate(u_r2s):
            u_flux[ii, jj] = flux[(r1s == r1) & (r2s == r2)].mean()

    fig = plt.figure()
    for ii, r1 in enumerate(u_r1s):
        if not np.isinf(r1):
            plt.semilogy((u_r2s - r1), u_flux[ii], label=r'$R_1 = %.01f$ м' % r1)
        else:
            plt.semilogy([-.7, 1.3], [u_flux[-1, -1]] * 2, '--', label=r'$R_1 = \infty$ м')

    fig.set_size_inches(w=fig_width, h=fig_width*fig_aspect2)
    plt.xlabel('$R_2 - R_1$, м')
    plt.ylabel('Поток на образце, [ф/с]')
    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    plt.legend(loc=7, bbox_to_anchor=(1.4, 0.5))
    plt.grid(alpha=.4)

    if save:
        fig.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/flux_vs_r1r2.pgf')


def tth_error():
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/SKIF_1_5/img/ae_scan'

    flux, offset = [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'ES' or metadata['axes'] != 'XZ':
            continue

        print(metadata)

        offset.append(metadata['tth_offset'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)

    flux, offset = np.array(flux), np.array(offset)
    ii = np.argsort(offset)
    flux, offset = flux[ii], offset[ii]

    fig = plt.figure()
    fig.set_size_inches(w=fig_width, h=fig_width * fig_aspect2)
    plt.plot(offset * 60., flux, '+-')
    plt.xlabel(r'$\theta - \theta_{B},\,[угл. сек.]$')
    plt.ylabel(r'Поток, [ф/с]')
    plt.title('Поток через разъюстированные кристаллы')
    plt.xlim(-.025 * 60, .025 * 60)
    plt.tight_layout()
    # R = 8 m, E = 150 keV, t = 1.2, alpha = 21.9

    if save:
        fig.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/tth_error.pgf')


if __name__ == '__main__':
    save = False

    if save:
        import matplotlib
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "pgf.preamble": "\n".join([
                r"\usepackage[T2A]{fontenc}", r"\usepackage[utf8]{inputenc}", r"\usepackage[english,russian]{babel}",
            ]),
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    # ##################################################################################################################
    # reflectivity_vs_t()
    # reflectivity_vs_r()
    # reflectivity_vs_chi()
    # flux_vs_e_chis()
    # flux_vs_e_ts()
    # flux_vs_r1r2()
    tth_error()
    # ##################################################################################################################
    if not save:
        plt.show()
