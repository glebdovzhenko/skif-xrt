import numpy as np
from matplotlib import pyplot as plt
import xrt.backends.raycing.sources as rsources

import params


if __name__ == '__main__':
    energy = np.logspace(1, 5 + np.log10(2.), 50)

    for bl in ('1_3', '1_5'):
        md = getattr(params, 'params_' + bl)
        wiggler_kwargs = getattr(params.sources, 'wiggler_' + bl + '_kwargs')

        theta = np.linspace(-md.front_end_h_angle, md.front_end_h_angle, 1000) / 2.
        psi = np.linspace(-md.front_end_v_angle, md.front_end_v_angle, 500) / 2.

        source = rsources.Wiggler(
            eMin=energy[0],
            eMax=energy[-1],
            distE='BW',
            xPrimeMax=theta[-1] * 1e3,
            zPrimeMax=psi[-1] * 1e3,
            **params.sources.ring_kwargs,
            **wiggler_kwargs
        )

        I0, l1, l2, l3 = source.intensities_on_mesh(energy, theta, psi)

        dtheta, dpsi = theta[1] - theta[0], psi[1] - psi[0]
        flux = I0.sum(axis=(1, 2)) * dtheta * dpsi

        s_data = np.loadtxt('spectra/wiggler_slit_%s.txt' % bl, skiprows=1, delimiter='\t')

        plt.figure(bl)
        plt.plot(energy * 1e-3, flux, label='xrt')
        plt.plot(s_data[:, 0] * 1e-3, s_data[:, 1], label='SPECTRA-11')
        plt.xlabel('Energy, [keV]')
        plt.ylabel('Flux, [ph/s / 0.1% BW]')
        plt.legend()

    plt.show()
