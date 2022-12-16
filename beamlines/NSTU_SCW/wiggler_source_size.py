import json
import numpy as np
import os
from matplotlib import pyplot as plt


if __name__ == '__main__':
    wd = 'img/SPECTRA'
    ks, x_fwhm, y_fwhm = [], [], []
    for f_name in filter(lambda x: '.json' in x, os.listdir(wd)):
        print(f_name)
        with open(os.path.join(wd, f_name), 'r') as f:
            j_obj = json.load(f)
            print(j_obj['Input']['Light Source']['K value'])

            xs = np.array(j_obj['Output']['data'][0])
            ys = np.array(j_obj['Output']['data'][1])
            fd = np.array(j_obj['Output']['data'][2]).reshape((*ys.shape, *xs.shape))

            fd_total_x = np.sum(fd, axis=0)
            fd_total_y = np.sum(fd, axis=1)

            fd_cs_x = fd[fd.shape[0] // 2, :]
            fd_cs_y = fd[:, fd.shape[1] // 2]

            ks.append(j_obj['Input']['Light Source']['K value'])

            x_fwhm.append(np.sqrt(np.sum(fd_cs_x * xs ** 2) / np.sum(fd_cs_x)))
            y_fwhm.append(np.sqrt(np.sum(fd_cs_y * ys ** 2) / np.sum(fd_cs_y)))

            plt.plot(xs, fd_cs_x)
            plt.plot(ys, fd_cs_y)
            plt.show()

    ks, x_fwhm, y_fwhm = np.array(ks), np.array(x_fwhm), np.array(y_fwhm)
    ii = np.argsort(ks)
    ks, x_fwhm, y_fwhm = ks[ii], x_fwhm[ii], y_fwhm[ii]
    plt.plot(ks, x_fwhm)
    plt.plot(ks, y_fwhm)
    plt.show()

