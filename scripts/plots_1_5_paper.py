import numpy as np

from matplotlib import pyplot as plt

from utils.calc import CrystalReflectivity
from utils.xrtutils import bell_fit


def reflectivity_vs_t():
    en = 30.e3  # eV
    alpha = np.radians(5.)
    crR = 2000  # mm
    crT = 2.  # mm
    thetas = 200e-6

    thetas = np.linspace(-0.5 * thetas, 1.5 * thetas, 10000)  # rad
    cr = CrystalReflectivity(useTT=True, R=crR)
    cr.t = crT

    figs, axes = [], []
    for _ in range(3):
        fig, ax = plt.subplots(1, 2)
        figs.append(fig)
        axes.extend(ax)

    for ii, en in enumerate([30.e3, 50.e3, 70.e3, 90.e3, 110.e3, 130.e3]):
        print(ii)
        for cr.t in [1., 1.5, 2., 2.5]:
            c_s, c_p = cr(thetas, en, alpha)
            ref = .5 * (np.abs(c_s) ** 2 + np.abs(c_p) ** 2)
            axes[ii].plot(thetas * 1e6, ref, label='$t = %.01f$ мм' % cr.t)

        axes[ii].set_ylim(0, 1.)
        axes[ii].set_xlabel(r'$\theta - \theta _B$, мкрад')
        axes[ii].set_ylabel(r'Коэффициент отражения')
        axes[ii].legend()
        axes[ii].set_title(r'$h \nu = %.01f$ кэВ' % (en * 1e-3))

    global save
    for ii, fig in enumerate(figs):
        fig.set_size_inches(w=5.39, h=5.39 * 9 / 16)
        fig.tight_layout()
        if save:
            fig.savefig('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/16_1-5_DCD_text/reflectivity_vs_t_%d.pgf' % ii)


if __name__ == '__main__':
    save = True

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
    reflectivity_vs_t()
    # ##################################################################################################################
    if not save:
        plt.show()
