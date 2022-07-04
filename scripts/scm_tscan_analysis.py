import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq


if __name__ == '__main__':
    ts = np.arange(0.5, 3.15, 0.05)
    fs = np.array(
        [2.01, 1.89, 1.49, 1.91, 1.93, 1.51, 1.86, 1.96, 1.54, 1.81,
         1.93, 1.57, 1.77, 1.93, 1.58, 1.73, 1.94, 1.61, 1.69, 1.94,
         1.64, 1.66, 1.91, 1.66, 1.63, 1.92, 1.71, 1.61, 1.87, 1.71,
         1.59, 1.85, 1.72, 1.59, 1.84, 1.76, 1.58, 1.80, 1.80, 1.58,
         1.77, 1.80, 1.59, 1.73, 1.81, 1.59, 1.72, 1.82, 1.58, 1.69,
         1.80, 1.59, 1.67]
    )
    fs_21m = np.array(
        [1.40, 1.35, 1.06, 1.37, 1.35, 1.08, 1.32, 1.37, 1.08, 1.29,
         1.38, 1.11, 1.24, 1.38, 1.12, 1.21, 1.36, 1.15, 1.19, 1.34,
         1.16, 1.17, 1.34, 1.18, 1.15, 1.34, 1.21, 1.15, 1.32]  # 1.90
    )

    data = np.load('../cmp_tests/crystals/result.npy')
    reduced = data.mean(axis=1)
    amin, amax = -30., 30.
    amin_s, amax_s = 21.0, 21.2

    ref_signal = reduced[int(reduced.shape[0] * (amin_s - amin) / (amax - amin)):
                         int(reduced.shape[0] * (amax_s - amin) / (amax - amin)), :].mean(axis=0)

    fig, ax = plt.subplots()
    ax.plot(rfftfreq(fs.shape[0], 0.05), np.abs(rfft(fs)) ** 2, '+--')
    ax.plot(rfftfreq(ref_signal.shape[0], 2. / ref_signal.shape[0]), np.abs(rfft(ref_signal)) ** 2, '+--')

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    ax1.plot(ts, fs)
    ax1.plot(ts[:29], fs_21m)
    ax3.plot(np.linspace(0.5, 2.5, ref_signal.shape[0]), ref_signal)

    im = ax2.imshow(reduced,
                    **{'extent': [0.5, 2.5, -30., 30.], 'aspect': 'auto', #'vmin': 0., 'vmax': 1.,
                       'origin': 'lower'}
                    )
    ax2.set_xlabel(r'thickness [mm]')
    ax2.set_ylabel(r'$\alpha$ (deg)')
    # fig.colorbar(im)

    plt.show()
