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

    plt.show()