import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq


if __name__ == '__main__':
    data = np.load('result.npy')
    print('α [-30, 30] deg, 2Θ-2Θ_0 [-5, 5] μrad, thickness [0.5, 2.5] mm', data.shape)

    reduced = data[450:550, 2700:2900, :].mean(axis=(0, 1))

    plt.subplot(121)
    plt.plot(np.arange(0.5, 2.5, 0.01), reduced)
    plt.subplot(122)
    plt.plot(rfftfreq(reduced.shape[0], 0.01), np.abs(rfft(reduced)) ** 2)
    plt.show()

    print('Period is ~0.16 mm')

    reduced = data.mean(axis=1)
    plt.imshow(reduced, **{'extent': [0.5, 2.5, -30., 30.],
                      'aspect': 'auto', 'vmin': 0., 'vmax': 1., 'origin': 'lower'})
    plt.xlabel(r'thickness [mm]')
    plt.ylabel(r'$\alpha$ (deg)')
    plt.colorbar()
    plt.show()

    # data = np.load('result_en.npy')
    # print('α [-30, 30] deg, 2Θ-2Θ_0 [-5, 5] μrad ([-1.031, 1.031] arcsec), en [25, 125] keV', data.shape)
    #
    # reduced = data[450:550, 2700:2900, :].mean(axis=(0, 1))
    #
    # for ii in range(data.shape[2]):
    #     plt.imshow(data[..., ii],
    #                **{'extent': [np.degrees(-5.e-6) * 3600., np.degrees(5.e-6) * 3600., -30., 30.],
    #                   'aspect': 'auto', 'vmin': 0., 'vmax': 1., 'origin': 'lower'})
    #     plt.xlabel(r'$\theta-\theta_B$ (arcsec)')
    #     plt.ylabel(r'$\alpha$ (deg)')
    #     plt.title(r'$h\nu$ = %.01f keV' % np.arange(25., 125., .5)[ii])
    #     plt.colorbar()
    #     plt.show()
    #
    # plt.subplot(121)
    # plt.plot(np.arange(25., 125., .5), reduced)
    # plt.subplot(122)
    # plt.plot(rfftfreq(reduced.shape[0], 1.), np.abs(rfft(reduced)) ** 2)
    # plt.show()
