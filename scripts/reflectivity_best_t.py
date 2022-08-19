import numpy as np
from matplotlib import pyplot as plt
import xrt.backends.raycing.materials as rm

from utils.calc import CrystalReflectivity

from params.params_1_5 import front_end_v_angle


if __name__ == '__main__':
    alpha = -np.radians(45.)
    crR = 50000  # mm

    thetas = np.linspace(-2.9 * front_end_v_angle, 0.1 * front_end_v_angle, 10000)
    cr = CrystalReflectivity(useTT=True)

    ts = np.linspace(.1, 4., 50)
    ens = np.arange(30., 120., 5.) * 1e3

    result = np.zeros(shape=(ts.shape[0], ens.shape[0]))

    for ii, en in enumerate(ens):
        for jj, t in enumerate(ts):
            print(ii, jj)
            cr.t = t
            cr.R = 1e3 * t

            c_s, c_p = cr(thetas, en, alpha)
            ref = .5 * (np.abs(c_s) ** 2 + np.abs(c_p) ** 2)
            plt.plot(thetas * 1e6, ref)

            result[jj, ii] = np.mean(ref)
        np.save('ref_t_en_alp45.npy', result)
