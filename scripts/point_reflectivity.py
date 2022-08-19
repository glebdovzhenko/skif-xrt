import numpy as np
from matplotlib import pyplot as plt
import xrt.backends.raycing.materials as rm

from utils.calc import CrystalReflectivity

from params.params_1_5 import front_end_v_angle


if __name__ == '__main__':
    en = 30.e3  # eV
    alpha = -np.radians(5.)
    crR = 50000  # mm

    thetas = np.linspace(-2.9 * front_end_v_angle, 0.1 * front_end_v_angle, 10000)
    cr = CrystalReflectivity(useTT=True, R=crR)
    t1, t2 = .1, 3.

    ts = np.linspace(t1, t2, 100)
    res = np.zeros(ts.shape)
    plt.figure()
    for ii, t in enumerate(ts):
        print(ii)
        cr.t = t
        cr.R = 1e3 * t

        c_s, c_p = cr(thetas, en, alpha)
        ref = .5 * (np.abs(c_s) ** 2 + np.abs(c_p) ** 2)
        plt.plot(thetas * 1e6, ref)
        res[ii] = np.mean(ref)
    plt.figure()
    plt.plot(ts, res)
    plt.show()

    # res = np.zeros((10, 20))
    #
    # for jj, crR in enumerate(np.linspace(1., 6., 10) * 1e3):
    #     plt.figure("%.01f" % crR)
    #     for ii, t in enumerate(np.linspace(t1, t2, 20)):
    #         print(crR * 1e-3, t)
    #
    #         cr.R = crR
    #         cr.t = t
    #         c_s, c_p = cr(thetas, en, alpha)
    #
    #         ref = .5 * (np.abs(c_s) ** 2 + np.abs(c_p) ** 2)
    #
    #         plt.plot(thetas * 1e6, ref, label='$t=%d$ mcm' % int(t * 1e3))
    #         res[jj, ii] = np.mean(ref)
    #
    #     plt.ylim(0, 1)
    #     plt.xlabel(r'$\theta-\theta_B$, mcrad')
    #     plt.ylabel(r'Reflectivity')
    #     plt.legend()
    #
    # plt.figure()
    # plt.imshow(res)
    # plt.show()
