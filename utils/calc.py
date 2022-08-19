import numpy as np
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.myopencl as mcl
import math
from scipy.signal import argrelextrema
from scipy.optimize import minimize


def chukhovskii_krisch_r2(r1, theta, c1chi, c2chi, source_c1_y, c1_c2_z, source_f2_y=0.):
    """
    Calculates bending radius for the 2nd crystal from the bending radius for the 1st crystal and the beamline params
    :return:
    """
    a = 4. * source_c1_y * np.abs(np.cos(c1chi - theta)) * (np.cos(c2chi + theta) ** 2 * (
                source_c1_y - source_f2_y + c1_c2_z / np.tan(2. * theta)) + c1_c2_z * np.cos(
        c2chi - theta) ** 2 / np.sin(2. * theta)) + 2. * source_c1_y * np.cos(c1chi + theta) * (
                    2. * (source_c1_y - source_f2_y) * np.cos(c2chi + theta) ** 2 + c1_c2_z * (
                        1. / np.tan(theta) + np.cos(2. * c2chi - theta) / np.sin(theta) - np.sin(2. * (c2chi + theta))))
    b = 4. * source_c1_y * np.cos(c1chi - theta) ** 2 * np.cos(c2chi - theta) ** 2 - 2. * np.cos(c1chi + theta) ** 2 * (
                2. * (source_c1_y - source_f2_y) * np.cos(c2chi + theta) ** 2 + c1_c2_z * (
            1. / np.tan(theta) + np.cos(2. * c2chi - theta) / np.sin(theta) - np.sin(2. * (c2chi + theta))))
    c = 4. * source_c1_y * np.cos(c1chi - theta) ** 2 * np.cos(c2chi + theta) * (
                source_c1_y - source_f2_y + c1_c2_z / np.tan(2. * theta)) - 4. * source_c1_y * np.abs(
        np.cos(c1chi - theta)) * np.cos(c2chi + theta) ** 2 * (
                    source_c1_y - source_f2_y + c1_c2_z / np.tan(2. * theta)) - 4. * source_c1_y * np.cos(
        c1chi + theta) * np.cos(c2chi + theta) ** 2 * (
                    source_c1_y - source_f2_y + c1_c2_z / np.tan(2. * theta)) - 4. * source_c1_y * c1_c2_z * np.abs(
        np.cos(c1chi - theta)) * np.cos(c2chi - theta) ** 2 / np.sin(2. * theta) - 4. * source_c1_y * c1_c2_z * np.cos(
        c2chi - theta) ** 2 * np.cos(c1chi + theta) / np.sin(2. * theta) - 4. * c1_c2_z * np.cos(
        c1chi + theta) ** 2 * np.cos(c2chi + theta) * (
                    source_c1_y - source_f2_y + c1_c2_z / np.tan(2. * theta)) / np.sin(2. * theta) + .25 * np.abs(
        np.cos(c2chi - theta)) * (1. / np.sin(theta)) ** 2 * (1. / np.cos(theta)) ** 2 * (
                    -2. * c1_c2_z - 2. * c1_c2_z * np.cos(2. * (c1chi + theta)) + source_c1_y * np.sin(
                2. * c1chi) - source_c1_y * np.sin(2. * c1chi - 4. * theta) + 2. * source_c1_y * np.sin(2. * theta)) * (
        c1_c2_z * np.cos(2. * theta) + (source_c1_y - source_f2_y) * np.sin(2. * theta))
    d = -4. * source_c1_y * np.cos(c1chi - theta) ** 2 * np.cos(c2chi - theta) ** 2 + 4. * np.cos(
        c1chi + theta) ** 2 * np.cos(c2chi + theta) ** 2 * (
                    source_c1_y - source_f2_y + c1_c2_z / np.tan(2 * theta)) + 4. * c1_c2_z * np.cos(
        c2chi - theta) ** 2 * np.cos(c1chi + theta) ** 2 / np.sin(2. * theta)
    e = 4. * source_c1_y * c1_c2_z * np.abs(np.cos(c1chi - theta)) * np.cos(c2chi + theta) * (
                source_c1_y - source_f2_y + c1_c2_z / np.tan(2. * theta)) / np.sin(
        2. * theta) + 4. * source_c1_y * c1_c2_z * np.cos(c1chi + theta) * np.cos(c2chi + theta) * (
                    source_c1_y - source_f2_y + c1_c2_z / np.tan(2. * theta)) / np.sin(
        2. * theta) + source_c1_y * c1_c2_z * np.abs(np.cos(c1chi - theta)) * np.abs(np.cos(c2chi - theta)) * (
                    np.cos(theta) * np.sin(theta)) ** -2 * (
                    c1_c2_z * np.cos(2. * theta) + (source_c1_y - source_f2_y) * np.sin(
                2. * theta)) + source_c1_y * c1_c2_z * np.abs(np.cos(c2chi - theta)) * np.cos(c1chi + theta) * (
                    np.cos(theta) * np.sin(theta)) ** -2 * (
                    c1_c2_z * np.cos(2. * theta) + (source_c1_y - source_f2_y) * np.sin(2. * theta))

    return r1 + (c * r1 + d * r1 ** 2 + e) / (a + b * r1)


class CrystalReflectivity:
    def __init__(self, geom='Laue reflected', hkl=(1, 1, 1), t=1., factDW=1., useTT=False, R=np.inf):
        self.cr = rm.CrystalSi(geom=geom, hkl=hkl, t=t, factDW=factDW)

        if useTT:
            self.matCL = mcl.XRT_CL(r'materials.cl', targetOpenCL='CPU')
            self.R = R
        else:
            self.matCL = None
            self.R = None

    @property
    def t(self):
        return self.cr.t

    @property
    def d(self):
        return self.cr.d

    @t.setter
    def t(self, val):
        self.cr.t = val

    def __call__(self, thetas, energy, alpha):
        plane_norm = (0., math.sin(alpha), math.cos(alpha))
        surface_norm = (0., -1., 0.)

        theta0 = np.arcsin(rm.ch / (2 * self.cr.d * energy))
        k_inc = (np.zeros_like(thetas), np.cos(thetas + theta0 + alpha), -np.sin(thetas + theta0 + alpha))
        k_ref = (np.zeros_like(thetas), np.cos(thetas + theta0 - alpha), np.sin(thetas + theta0 - alpha))
        if self.matCL is not None:
            return self.cr.get_amplitude_TT(
                E=energy * np.ones_like(thetas),
                beamInDotNormal=np.dot(surface_norm, k_inc),
                beamOutDotNormal=np.dot(surface_norm, k_ref),
                beamInDotHNormal=np.dot(plane_norm, k_inc),
                ucl=self.matCL,
                alphaAsym=alpha,
                Rcurvmm=-self.R
            )
        else:
            return self.cr.get_amplitude(
                E=energy * np.ones_like(thetas),
                beamInDotNormal=np.dot(surface_norm, k_inc),
                beamOutDotNormal=np.dot(surface_norm, k_ref),
                beamInDotHNormal=np.dot(plane_norm, k_inc),
            )


def max_reflectivity_t(energy, alpha, harmonic_num, dTh=100.e-6, nTh=5600):
    result = np.zeros(shape=(1000,))
    cr = CrystalReflectivity()

    ts = np.linspace(.1, 5., result.shape[0])  # mm
    thetas = np.linspace(-dTh, dTh, nTh)

    for ii, cr.t in enumerate(ts):
        cur_s, cur_p = cr(thetas, energy, alpha)
        result[ii] = .5 * np.mean(np.abs(cur_s) ** 2 + np.abs(cur_p) ** 2)

    minimums = ts[argrelextrema(result, np.less)]
    maximums = ts[argrelextrema(result, np.greater)]

    if minimums[0] > maximums[0]:
        minimums = np.insert(minimums, 0, ts[0])[:-1]

    def f(x, *args):
        cr.t = x
        cur_s, cur_p = cr(thetas, energy, alpha)
        return -.5 * np.mean(np.abs(cur_s) ** 2 + np.abs(cur_p) ** 2)

    for ii, (lb, rb, x0) in enumerate(zip(minimums[:-1], minimums[1:], maximums[:-1])):
        if ii != harmonic_num - 1:
            continue

        # from matplotlib import pyplot as plt
        # plt.plot(ts, result)
        # plt.show()
        return minimize(f, x0, bounds=((lb, rb),)).x


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # r1s = np.linspace(1000., 100000., 1000)
    # r2s = chukhovskii_krisch_r2(r1s,
    #                             theta=0.02824848223580159,
    #                             c1chi=np.radians(1.),
    #                             c2chi=np.radians(1.),
    #                             source_c1_y=33500.,
    #                             c1_c2_z=25.)
    # xs = np.array(
    #     [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000,
    #      17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000,
    #      32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000, 41000, 42000, 43000, 44000, 45000, 46000,
    #      47000, 48000, 49000, 50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 60000])
    # ys = np.array([1857.77, 2834.84, 3812.56, 4790.92, 5769.92, 6749.57, 7729.87, 8710.81, 9692.41, 10674.6, 11657.5,
    #                12641.1, 13625.3, 14610.1, 15595.6, 16581.8, 17568.6, 18556., 19544.1, 20532.9, 21522.3, 22512.4,
    #                23503.1,
    #                24494.5, 25486.6, 26479.3, 27472.7, 28466.7, 29461.4, 30456.8, 31452.8, 32449.5, 33446.8, 34444.8,
    #                35443.5, 36442.8, 37442.8, 38443.5, 39444.8, 40446.9, 41449.5, 42452.9, 43456.9, 44461.6, 45467.,
    #                46473.,
    #                47479.7, 48487.1, 49495.2, 50503.9, 51513.3, 52523.4, 53534.1, 54545.6, 55557.7, 56570.5, 57584.,
    #                58598.1,
    #                59613., 60628.5])
    # plt.plot(r1s, r2s - r1s)
    # plt.plot(xs, ys - xs, '+')

    print(max_reflectivity_t(30e3, np.radians(30.), 1, dTh=100.e-6, nTh=5600))
    plt.show()