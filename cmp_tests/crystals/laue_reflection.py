import numpy as np

import xrt.backends.raycing.materials as rm
import math

from matplotlib import pyplot as plt


def rocking_curve(t=.1, energy=10000):
    cr = rm.CrystalSi(
        geom='Laue reflected',
        hkl=(1, 1, 1),
        d=3.13562,
        t=t,
        factDW=1.
    )

    def get_amplitude(alpha):
        alpha = math.radians(alpha)
        theta0 = np.arcsin(rm.ch / (2 * cr.d * energy))
        theta = theta0 + np.linspace(-5, 5, 5600) * 1e-6

        surface_norm = (0., -1., 0.)
        plane_norm = (0., math.sin(alpha), math.cos(alpha))
        k_inc = (np.zeros_like(theta), np.cos(theta + alpha), -np.sin(theta + alpha))
        k_ref = (np.zeros_like(theta), np.cos(theta - alpha), np.sin(theta - alpha))

        cur_s, cur_p = cr.get_amplitude(
            E=energy * np.ones_like(theta),
            beamInDotNormal=np.dot(surface_norm, k_inc),
            beamOutDotNormal=np.dot(surface_norm, k_ref),
            beamInDotHNormal=np.dot(plane_norm, k_inc)
        )

        return np.degrees(theta - theta0), cur_s, cur_p

    cs_data, cp_data, alphas, thetas = [], [], np.linspace(-30., 30., 1000), []
    for alpha in alphas:
        th, c_s, c_p = get_amplitude(alpha)
        # fig = plt.figure()
        # fig.suptitle(r'Laue reflectivity: $\alpha$ = %.2f$^{\circ}$, t = %.f $\mu$m, $h\nu$ = %.f keV' %
        #              (alpha, t * 1e3, energy * 1e-3))
        # plt.plot(np.array(th) * 3600, np.abs(c_s) ** 2, label=r'$R_{\sigma}$')
        # plt.plot(np.array(th) * 3600, np.abs(c_p) ** 2, label=r'$R_{\pi}$')
        # plt.plot(np.array(th) * 3600, .5 * (np.abs(c_s) ** 2 + np.abs(c_p) ** 2), label=r'$R_{\sigma} + R_{\pi}$')
        # plt.ylim((0, 1))
        # plt.show()
        cp_data.append(c_p)
        cs_data.append(c_s)
        thetas = th

    cs_data, cp_data, thetas = np.array(cs_data), np.array(cp_data), np.array(thetas) * 3600

    plt_kwargs = {
        'extent': [np.min(thetas), np.max(thetas), np.min(alphas), np.max(alphas)],
        'aspect': 'auto',
        'vmin': 0.,
        'vmax': 1.,
        'origin': 'lower'
    }

    fig, axd = plt.subplot_mosaic(
        mosaic=[['s', 'p', 's+p'], ['cb', 'cb', 'cb']],
        gridspec_kw={
            "height_ratios": [7, 1],
            "width_ratios": [1, 1, 1],
        },
        constrained_layout=True
    )

    fig.suptitle(r'Laue reflectivity: t = %.f $\mu$m, $h\nu$ = %.f keV' % (t * 1e3, energy * 1e-3))

    axd['s'].set_title(r'$\sigma$-polarization')
    axd['s'].imshow(
        X=np.abs(cs_data) ** 2,
        **plt_kwargs
    )
    axd['s'].set_xlabel(r'$\theta-\theta_B$ (arcsec)')
    axd['s'].set_ylabel(r'$\alpha$ (deg)')

    axd['p'].set_title(r'$\pi$-polarization')
    axd['p'].imshow(
        X=np.abs(cp_data) ** 2,
        **plt_kwargs
    )
    axd['p'].set_xlabel(r'$\theta-\theta_B$ (arcsec)')
    axd['p'].set_ylabel(r'$\alpha$ (deg)')

    axd['s+p'].set_title(r'$R_{\pi} + R_{\sigma}$')
    ax2 = axd['s+p'].imshow(
        X=0.5 * (np.abs(cp_data) ** 2 + np.abs(cs_data) ** 2),
        **plt_kwargs
    )
    axd['s+p'].set_xlabel(r'$\theta-\theta_B$ (arcsec)')
    axd['s+p'].set_ylabel(r'$\alpha$ (deg)')

    print('At %.f μm, %.f keV: max Rs, Rp: %.03f, %.03f' % (
        t * 1e3, energy * 1e-3, np.max(np.abs(cp_data) ** 2), np.max(np.abs(cs_data) ** 2)
    ))

    axd['s'].sharex(axd['p'])
    axd['p'].sharex(axd['s+p'])
    axd['s'].sharey(axd['p'])
    axd['p'].sharey(axd['s+p'])
    fig.colorbar(ax2, cax=axd['cb'], orientation='horizontal')

    return 0.5 * (np.abs(cp_data) ** 2 + np.abs(cs_data) ** 2)


if __name__ == '__main__':
    try:
        result = []
        for tt in np.arange(0.5, 2.5, 0.01):
            result.append(rocking_curve(t=tt, energy=70000))
            plt.show()
        result = np.stack(result, axis=-1)
    except KeyboardInterrupt:
        plt.show()
    else:
        np.save(file='result.npy', arr=result)
        plt.show()
