import pickle

import numpy as np
import os

import xrt.backends.raycing.materials as rm
import math

# import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "pgf.preamble": "\n".join([
#         r"\usepackage[T2A]{fontenc}",
#         r"\usepackage[utf8]{inputenc}",
#         r"\usepackage[english,russian]{babel}",
#     ]),
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

from matplotlib import pyplot as plt


ts_alp1 = np.array([
    0.23, 0.40, 0.57, 0.73, 0.90, 1.07, 1.23, 1.40, 1.57, 1.73, 1.90, 2.07, 2.23, 2.40
])
fs_alp1 = np.array([
    2.25, 2.13, 2.06, 2.02, 1.99, 1.95, 1.95, 1.91, 1.89, 1.89, 1.88, 1.87, 1.86, 1.84
])
ts_alp29 = np.array([
    0.20, 0.35, 0.49, 0.64, 0.79, 0.93, 1.08, 1.22, 1.37, 1.51, 1.66, 1.81, 1.95, 2.10  # , 2.24, 2.38
])
fs_alp29 = np.array([
    2.30, 2.14, 2.11, 2.07, 2.01, 1.99, 1.97, 1.95, 1.95, 1.92, 1.91, 1.90, 1.89, 1.87  # , 1.86, 1.85
])


def set_global_vars():

    global cr_bend_rs1, cr_bend_fs1, \
        cr_bend_rs2, cr_bend_fs2, cr_bend_en_br2, cr_bend_zpr_br2, \
        cr_bend_rs3, cr_bend_fs3

    def get_data(dd):
        rad, flux, zpr_br, en_br = [], [], [], []
        for f_name in filter(lambda x: ('SCM Slit Directions-' in x) and ('.pickle' in x), os.listdir(dd)):
            with open(os.path.join(dd, f_name), 'rb') as f:
                rad.append(float(f_name[f_name.find('-') + 1:f_name.find('m.')]))
                data = pickle.load(f)
                flux.append(data.flux)

                bins = .5 * data.ybinEdges[1:] + .5 * data.ybinEdges[:-1]
                zpr_br.append(
                    np.sum((.5 * data.ytotal1D[1:] + .5 * data.ytotal1D[:-1]) * (bins[1:] - bins[:-1])) /
                    np.max(data.ytotal1D)
                )

                bins = .5 * data.ebinEdges[1:] + .5 * data.ebinEdges[:-1]
                en_br.append(
                    np.sum((.5 * data.etotal1D[1:] + .5 * data.etotal1D[:-1]) * (bins[1:] - bins[:-1])) /
                    np.max(data.etotal1D)
                )

        rad, flux, zpr_br, en_br = np.array(rad), np.array(flux), np.array(zpr_br), np.array(en_br)
        ii = np.argsort(rad)

        return rad[ii], flux[ii], zpr_br[ii], en_br[ii]

    cr_bend_rs2, cr_bend_fs2, cr_bend_zpr_br2, cr_bend_en_br2 = get_data(
        '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/img/70keV-Rscan'
    )

    cr_bend_rs1, cr_bend_fs1, _, _ = get_data(
        '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/img/70keV-Rscan-mono'
    )

    cr_bend_rs3, cr_bend_fs3, _, _ = get_data(
        '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/beamlines/img/70keV-Rscan-mono-alp29'
    )


def get_amplitude(alpha, energy, t, theta_inc):
    cr = rm.CrystalSi(
        geom='Laue reflected',
        hkl=(1, 1, 1),
        d=3.13562,
        t=t,
        factDW=1.
    )

    alpha = math.radians(alpha)
    theta0 = np.arcsin(rm.ch / (2 * cr.d * energy))

    theta_inc += theta0

    surface_norm = (0., -1., 0.)
    plane_norm = (0., math.sin(alpha), math.cos(alpha))
    k_inc = (np.zeros_like(theta_inc), np.cos(theta_inc + alpha), -np.sin(theta_inc + alpha))
    k_ref = (np.zeros_like(theta_inc), np.cos(theta_inc - alpha), np.sin(theta_inc - alpha))

    cur_s, cur_p = cr.get_amplitude(
        E=energy * np.ones_like(theta_inc),
        beamInDotNormal=np.dot(surface_norm, k_inc),
        beamOutDotNormal=np.dot(surface_norm, k_ref),
        beamInDotHNormal=np.dot(plane_norm, k_inc)
    )

    fig = plt.figure()
    fig.suptitle(r'Коэффициент отражения: $\chi$ = %.2f$^{\circ}$, t = %.f мкм, $h\nu$ = %.f кэВ' %
                 (math.degrees(alpha), t * 1e3, energy * 1e-3))
    plt.plot((theta_inc - theta0) * 1e3, np.abs(cur_s) ** 2, label=r'$R_{\sigma}^2$')
    plt.plot((theta_inc - theta0) * 1e3, np.abs(cur_p) ** 2, label=r'$R_{\pi}^2$')
    plt.ylim((0, 1))
    plt.xlabel(r'$\theta-\theta_B$ (мрад)')
    plt.ylabel(r'$R^2$')
    plt.legend()

    fig.set_size_inches(w=5.39, h=5.39 * 9 / 16)
    plt.tight_layout()


def t_alpha_scan(energy, ts, alphas):
    result = np.zeros(shape=(ts.shape[0], alphas.shape[0]))
    cr = rm.CrystalSi(
        geom='Laue reflected',
        hkl=(1, 1, 1),
        d=3.13562,
        t=1.1,
        factDW=1.
    )

    thetas = np.linspace(-20, 20, 1000) * 1e-6
    theta0 = np.arcsin(rm.ch / (2 * cr.d * energy))
    thetas += theta0

    surface_norm = (0., -1., 0.)

    for ii, t in enumerate(ts):
        cr = rm.CrystalSi(
            geom='Laue reflected',
            hkl=(1, 1, 1),
            d=3.13562,
            t=t,
            factDW=1.
        )
        print(t)
        for jj, alpha in enumerate(alphas):
            alpha = math.radians(alpha)

            plane_norm = (0., math.sin(alpha), math.cos(alpha))
            k_inc = (np.zeros_like(thetas), np.cos(thetas + alpha), -np.sin(thetas + alpha))
            k_ref = (np.zeros_like(thetas), np.cos(thetas - alpha), np.sin(thetas - alpha))

            cur_s, cur_p = cr.get_amplitude(
                E=energy * np.ones_like(thetas),
                beamInDotNormal=np.dot(surface_norm, k_inc),
                beamOutDotNormal=np.dot(surface_norm, k_ref),
                beamInDotHNormal=np.dot(plane_norm, k_inc)
            )

            result[ii, jj] = np.mean(.5 * np.abs(cur_s)**2 + .5 * np.abs(cur_p)**2)

    fig = plt.figure()

    plt.imshow(result.T, **{
        'extent': [np.min(ts), np.max(ts), np.min(alphas), np.max(alphas)],
        'aspect': 'auto', 'origin': 'lower', 'cmap': 'turbo',
        #'vmin': 0, 'vmax': 0.1
    })

    plt.plot(ts_alp1, np.ones_like(ts_alp1), '+')
    plt.plot(ts_alp29, 29 * np.ones_like(ts_alp29), '+')

    plt.xlabel(r'$t$, мм')
    plt.ylabel(r'$\chi, \, ^{\circ}$')
    plt.title(r'Интегральный коэффициент отражения')
    plt.colorbar()
    fig.set_size_inches(w=5.39, h=5.39 * 9 / 16)
    plt.tight_layout()


def plot_int_f():
    fig = plt.figure()
    plt.plot(ts_alp1, fs_alp1, 'o-', label=r'$\chi = 1^{\circ}$')
    plt.plot(ts_alp29, fs_alp29, 'o-', label=r'$\chi = 29^{\circ}$')
    plt.xlabel(r'$t$, мм')
    plt.ylabel(r'$\Phi, \, 10^{13}$ф/с')
    plt.title(r'Интегральная интенсивность отражённого пучка')
    plt.legend()
    fig.set_size_inches(w=5.39, h=5.39 * 9 / 16)
    plt.grid()
    plt.tight_layout()


def r_scan_1cr():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, ax1 = plt.subplots()
    plt.title(r'Интегральная интенсивность отражённого пучка')

    color = colors[0]

    ax1.plot(cr_bend_rs1, cr_bend_fs1 / 1e12, label=r'$h \nu = 70 \pm 0.001$ кэВ', color=color)
    ax1.plot([33.5, 33.5], [0, 1.1e-12 * cr_bend_fs1.max()], '--', label=r'Расстояние до источника', color=color)
    ax1.set_ylabel(r'$\Phi, \, 10^{12}$ф/с', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend(loc='upper right')

    ax2 = ax1.twinx()

    color = colors[1]

    ax2.plot(cr_bend_rs2, cr_bend_fs2 / 1e13, label=r'$h \nu = 70 \pm 1$ кэВ', color=color)
    ax2.set_xlabel(r'Радиус изгиба, м')
    ax1.set_xlabel(r'Радиус изгиба, м')
    ax2.set_ylabel(r'$\Phi, \, 10^{13}$ф/с', color=color)
    ax2.set_ylim(2.7, 3.5)
    ax2.set_xlim(0, 90)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.set_size_inches(w=5.39, h=5.39 * 9 / 16)
    plt.legend(loc='upper left')
    plt.tight_layout()


def r_scan_1cr_alp():
    fig = plt.figure()
    plt.title(r'Интегральная интенсивность отражённого пучка')
    plt.plot(cr_bend_rs1, cr_bend_fs1 / 1e12, label=r'$\chi = 1 ^{\circ}$')
    plt.plot(cr_bend_rs3, cr_bend_fs3 / 1e12, label=r'$\chi = 29 ^{\circ}$')
    plt.plot([33.5, 33.5], [0, 1.1e-12 * max(cr_bend_fs1.max(), cr_bend_fs3.max())], '--', label=r'Расстояние до источника')
    plt.ylabel(r'$\Phi, \, 10^{12}$ф/с')
    plt.xlabel(r'Радиус изгиба, м')
    plt.xlim(0, 90)

    fig.set_size_inches(w=5.39, h=5.39 * 9 / 16)
    plt.legend(loc='upper right')
    plt.tight_layout()


def r_scan_1cr_widths():
    zpr_br = 1e3 * np.arctan(cr_bend_zpr_br2)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, ax1 = plt.subplots()
    plt.suptitle(r'Параметры отражённого пучка')

    color = colors[0]
    ax1.plot(cr_bend_rs2, cr_bend_en_br2, color=color, label=r'$\Delta E (R)$')
    ax1.plot([.9 * cr_bend_rs2[cr_bend_rs2 < np.inf].max(), 1.2 * cr_bend_rs2[cr_bend_rs2 < np.inf].max()],
             [cr_bend_en_br2[cr_bend_rs2 == np.inf][0]] * 2, '--',
             color=color, label=r'$\Delta E (R = \infty)$')
    ax1.set_ylabel(r'Ширина полосы пропускания $E$, эВ', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper center')

    ax2 = ax1.twinx()

    color = colors[1]

    ax2.plot(cr_bend_rs2, zpr_br, color=color, label=r'$\Delta \theta (R)$')
    ax2.plot([.9 * cr_bend_rs2[cr_bend_rs2 < np.inf].max(), 1.2 * cr_bend_rs2[cr_bend_rs2 < np.inf].max()],
             [zpr_br[cr_bend_rs2 == np.inf][0]] * 2, '--',
             color=color, label=r'$\Delta \theta (R = \infty)$')
    ax2.set_xlabel(r'Радиус изгиба, м')
    ax1.set_xlabel(r'Радиус изгиба, м')
    ax2.set_ylabel(r'Ширина полосы пропускания $\theta$, мрад', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    fig.set_size_inches(w=5.39, h=5.39 * 9 / 16)
    plt.tight_layout()


if __name__ == '__main__':
    set_global_vars()
    # get_amplitude(29.6, 70000, 2.24, np.linspace(-10, 10, 10000) * 1e-6)
    # plt.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/cr-ref.pgf')
    # t_alpha_scan(70000, np.arange(.1, 4.1, .01), np.arange(-45., 45., .5))
    # plt.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/mean-r.pgf')
    # plot_int_f()
    # plt.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/f-1cr-rtr.pgf')
    # r_scan_1cr()
    # plt.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/r-scan-1cr.pgf')
    # r_scan_1cr_alp()
    # plt.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/r-scan-1cr-2alp.pgf')
    r_scan_1cr_widths()
    # plt.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/r-scan-1cr-widths.pgf')
    plt.show()
