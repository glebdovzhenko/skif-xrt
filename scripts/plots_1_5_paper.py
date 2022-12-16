import numpy as np
import pickle
import os

from matplotlib import pyplot as plt

from utils.xrtutils import bell_fit, get_integral_breadth, get_line_kb

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


def meridional_focusing_c1():
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/PLAYGROUND/img/scan3'

    rs, f_dist, flux, br = [], [], [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'C1' or metadata['axes'] != 'ZZpr':
            continue

        rs.append(metadata['r1'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)
            k, b = get_line_kb(data, show=False)
            f_dist.append(-np.sign(k) * np.sqrt((1. / k) ** 2 + (b / k) ** 2))
            br.append(get_integral_breadth(data, axis='y'))
            print('R = %f' % metadata['r1'])
            print('Focus F = %f, 2Θ = %f' % (f_dist[-1], np.degrees(np.arctan(b))))

    rs, f_dist, flux, br = np.array(rs), np.array(f_dist), np.array(flux), np.array(br)
    ii = np.argsort(rs)
    rs, f_dist, flux, br = rs[ii], f_dist[ii] * 1e-3, flux[ii], br[ii]
    
    fig, ax = plt.subplots()
    fig.set_size_inches(w=fig_width, h=fig_width * fig_aspect2)

    ax.plot(rs, f_dist, label='Cr-to-F', color='C0')
    ax.plot([rs[1], rs[-2]], [f_dist[-1], f_dist[-1]], '--', label=r'Cr-to-F $R=\pm\infty$', color='C0')
    # ax.plot([rs[1], rs[-2]], [f_dist[0], f_dist[0]], '--', label=r'Cr-to-F $R=-\infty$', color='C0')
    ax.set_ylim(-50, 150)
    plt.legend(loc='upper left')
    plt.grid(alpha=.4)


    secax = ax.twinx()
    secax.plot(rs, br, label='Z\' Breadth', color='C1')
    plt.legend(loc='upper right')
    
    plt.xlabel('R, [m]')
    ax.set_ylabel('F, [м]', color='C0')
    secax.set_ylabel('Z\' FWHM, рад', color='C1')
    ax.tick_params(axis='y', labelcolor='C0')
    secax.tick_params(axis='y', labelcolor='C1')

    plt.title('1 кристалл')
    plt.tight_layout()

    if save:
        fig.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/merid_f_c1.pgf')


def meridional_focusing_c2():
    dd = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/PLAYGROUND/img/scan3'
    
    #################### SKIF 1-5 ####################
    ##### Superconducting Wiggler at [0, 0, 0]
    ##### Front End Slit at [0, 15000, 0]
    ##### Si[111] Crystal 1 at [0.0, 33500, 0.0]
    ##### Crystal 1 Monitor at [0.0, 33594.21541931476, 12.5]
    ##### Si[111] Crystal 2 at [0.0, 33688.43083862951, 25]
    ##### Crystal 2 Monitor at [0, 49990, 25]
    ##################################################

    rs, f_dist, flux, br = [], [], [], []
    for metadata in datafiles(dd):
        if metadata['name'] != 'C2' or metadata['axes'] != 'ZZpr':
            continue

        rs.append(metadata['r1'])

        with open(os.path.join(dd, metadata['file']), 'rb') as f:
            data = pickle.load(f)
            flux.append(data.flux)
            k, b = get_line_kb(data, show=False)
            f_dist.append(-np.sign(k) * np.sqrt((1. / k) ** 2 + (b / k) ** 2))
            br.append(get_integral_breadth(data, axis='y'))
            print('R = %f' % metadata['r1'])
            print('Focus F = %f, 2Θ = %f' % (f_dist[-1], np.degrees(np.arctan(b))))

    rs, f_dist, flux, br = np.array(rs), np.array(f_dist), np.array(flux), np.array(br)
    ii = np.argsort(rs)
    rs, f_dist, flux, br = rs[ii], f_dist[ii] * 1e-3, flux[ii], br[ii]
    f_dist += 49.990 - 33.688
    
    fig, ax = plt.subplots()
    fig.set_size_inches(w=fig_width, h=fig_width * fig_aspect2)

    ax.plot(rs, f_dist, label='Cr-to-F', color='C0')
    ax.plot([rs[1], rs[-2]], [f_dist[-1], f_dist[-1]], '--', label=r'Cr-to-F $R=\pm\infty$', color='C0')
    ax.set_ylim(-35, -30)
    plt.legend(loc='upper left')
    plt.grid(alpha=.4)


    secax = ax.twinx()
    secax.plot(rs, br, label='Z\' Breadth', color='C1')
    plt.legend(loc='upper right')
    
    plt.xlabel('R, [m]')
    ax.set_ylabel('F, [м]', color='C0')
    secax.set_ylabel('Z\' FWHM, рад', color='C1')
    ax.tick_params(axis='y', labelcolor='C0')
    secax.tick_params(axis='y', labelcolor='C1')

    plt.title('2 кристалл')
    plt.tight_layout()

    if save:
        fig.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/merid_f_c2.pgf')


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
    # tth_error()
    meridional_focusing_c1()
    # meridional_focusing_c2()
    # ##################################################################################################################
    if not save:
        plt.show()
