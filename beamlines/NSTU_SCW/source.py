import numpy as np
from matplotlib import pyplot as plt
import xrt.backends.raycing.sources as rsources

import params


if __name__ == '__main__':
    # energy = np.logspace(1, 5 + np.log10(2.), 50)
    energy = np.linspace(49e3, 51e3, 50)

    bl = '1_5'
    md = getattr(params, 'params_' + bl)
    wiggler_kwargs = getattr(params.sources, 'wiggler_' + bl + '_kwargs')

    theta = np.linspace(-4e-3, 4e-3, 101)
    psi = np.linspace(-2e-4, 2e-4, 101)

    source = rsources.Wiggler(
        eMin=energy[0],
        eMax=energy[-1],
        distE='BW',
        xPrimeMax=theta[-1] * 1e3,
        zPrimeMax=psi[-1] * 1e3,
        **params.sources.ring_kwargs,
        **wiggler_kwargs
    )

    # #
    # # calculating angular profiles
    # #
    # I0, l1, l2, l3 = source.intensities_on_mesh(energy, theta, psi)
    #
    # dtheta, dpsi, de = theta[1] - theta[0], psi[1] - psi[0], energy[1] - energy[0]
    # plt_mesh = np.mean(I0, axis=0) * 1e-6  # rad^2 -> mrad^2 ???
    #
    # plt.figure()
    # plt.title(r'E = %.01f keV [ph / s / mrad$^2$ / 0.1%%BW]' % (np.mean(energy) * 1e-3))
    # plt.imshow(plt_mesh.T,
    #            extent=[theta[0] * 1e3, theta[-1] * 1e3, psi[0] * 1e3, psi[-1] * 1e3],
    #            aspect='auto')
    # plt.xlabel(r'Horizontal $\theta$, [mrad]')
    # plt.ylabel(r'Vertical $\psi$, [mrad]')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()


    # #
    # # calculating spacial profiles
    # # FWHM = 2. * np.sqrt(2. * np.log(2)) * sigma
    # #
    # flux, xpr, zpr, x, z, = source.real_photon_source_sizes(energy, theta, psi)
    #
    # print('Energy = %.01f keV' % (np.mean(energy) * 1e-3))
    # print('FWHM X = %f mm' % (2. * np.sqrt(2. * np.log(2)) * np.sqrt(np.mean(x))))
    # print('FWHM Z = %f mm' % (2. * np.sqrt(2. * np.log(2)) * np.sqrt(np.mean(z))))
    # print('FWHM X\' = %f rad' % (2. * np.sqrt(2. * np.log(2)) * np.sqrt(np.mean(xpr))))
    # print('FWHM Z\' = %f rad' % (2. * np.sqrt(2. * np.log(2)) * np.sqrt(np.mean(zpr))))


    #
    # calculating power vs K
    #
    Ks = np.linspace(1, 20, 8)
    theta = np.linspace(-1e-3, 1e-3, 21)
    psi = np.linspace(-1e-4, 1e-4, 21)
    energy = np.linspace(50e3, 51e3, 50)
    # source.n = 36
    plt.figure()
    plt.title('Wiggler power in 1 by 0.1 mrad')
    plt.xlabel('K')
    plt.ylabel('Power [W]')
    plt.plot(Ks, source.power_vs_K(energy, theta, psi, Ks))
    plt.show()

