
import numpy as np

from matplotlib import pyplot as plt


if __name__ == '__main__':
    en = 30.e3  # eV
    alpha = np.radians(38.)
    crR = 10000.0  # cm
    crT = .2  # cm
    thetas = np.linspace(-100, 300, 10000)  # μrad

    plt.figure()
    # plt.plot(thetas, ref, label='xrt')
    # plt.plot(thetas, ref, label='xrt')
    plt.xlabel(r'$\theta - \theta_B, \, [\mu \mathrm{rad}]$')
    plt.ylabel('Reflectivity')

    data = np.loadtxt('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/09_oasys/SKIF-1-5/wd/diff_pat.dat', skiprows=5)
    plt.plot(data[:, 0], .5 * (data[:, -1] + data[:, -2]), label='OASYS')
    plt.legend()
    plt.show()
