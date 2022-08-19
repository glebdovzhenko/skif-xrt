import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    for alp in (5, 15, 30, 45):
        data = np.load('ref_t_en_alp%d.npy' % alp)
        plt.figure('alpha = %d' % alp)
        plt.imshow(data, **{'extent': [27.5, 117.5, .1, 4.], 'aspect': 'auto', 'origin': 'lower', 'vmin': 0, 'vmax': .6})
        plt.colorbar()
        plt.xlabel('Energy, [keV]')
        plt.ylabel('Thickness, [mm] and Radius [m]')
    plt.show()