import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit
import os


class Normalization:
    def __init__(self):
        self.slope = []
        self.intercept = 0

    def calc(self, x_, y_, z_):
        reg = LinearRegression(fit_intercept=True).fit(np.array([x_, y_]).T, z_)

        self.slope = reg.coef_
        self.intercept = reg.intercept_

    def apply(self, x_, y_, z_, axis=1):
        x_r, y_r, z_r = x_.copy(), y_.copy(), z_.copy()

        z_r -= self.intercept
    
        if axis == 0:
            r = Rotation.from_rotvec(-np.arctan(self.slope[0]) * np.array([0., 1., 0.]))
        elif axis == 1:
            r = Rotation.from_rotvec(-np.arctan(self.slope[1]) * np.array([1., 0., 0.]))
        else:
            raise ValueError('axis can be only 0 or 1')
    
        x_r, y_r, z_r = r.apply(np.array([x_r, y_r, z_r]).T).T

        return x_r, y_r, z_r


def multigaussian(x_, K, A, C, C0):
    """
    Calculates multidimensional Gaussian distribution as
    C * (det[K / 2pi])^{-1/2} * exp{A.K^{-1}A / 2} * exp{-x.Kx / 2 + A.x}.
    If x is n-dimensional, C is a constant, A is n-dimensional, K is a n by n matrix.
    """
    if x_.ndim == 1:
        x_ = x_.reshape((1, ) + x_.shape)
    elif x_.ndim == 2:
        x_ = x_.reshape((x_.shape[0], 1, x_.shape[1]))
    else:
        raise ValueError('x_ arg can only be 1- or 2-dimensional')
        
    xKx = np.matmul(x_, np.matmul(K, np.swapaxes(x_, -1, -2))).flatten()
    Ax = np.matmul(A, np.swapaxes(x_, -1, -2)).flatten()
    AKm1A = np.matmul(A, np.matmul(np.linalg.inv(K), A.T))
    mgn = np.linalg.det(K / (2. * np.pi)) ** (.5)
      
    return C * (mgn / np.exp(.5 * AKm1A)) * np.exp(-.5 * xKx + Ax) + C0


def digaussian(x_, cx, cy, sx, sy, a):
    """
    exp{}
    """
    return a * np.exp(-.5 * ((x_[0] - cx) / sx)**2 - .5 * ((x_[1] - cy) / sy)**2)


if __name__ == '__main__':
    wd = r'/Users/glebdovzhenko/Downloads/Нагрузка на 1-й кристалл/exp_data'
    save_d = r'/Users/glebdovzhenko/Yandex.Disk.localized/Dev/skif-xrt/datasets/bump'
    
    for sd, _, fs in os.walk(wd):
        for fpath in filter(lambda x: x[-4:] == '.txt', fs):
            print(os.path.join(os.path.basename(sd), fpath))
            
            # reading data from file
            a = np.genfromtxt(os.path.join(sd, fpath), comments='%')
            x1, y1, z1, dx, dy, dz = a.T
            x0, y0, z0 = x1 - dx, y1 - dy, z1 - dz
            
            # rotating the surface parallel to XY plane
            norm = Normalization()
            for _ in range(2):
                norm.calc(x0, y0, z0)
                x0, y0, z0 = norm.apply(x0, y0, z0, axis=1)
                x1, y1, z1 = norm.apply(x1, y1, z1, axis=1)
            
            # defining a digaussian
            def dg(arg, *args):
                return digaussian(
                    arg,
                    cx=args[0],
                    cy=args[1],
                    sx=args[2],
                    sy=args[3],
                    a=args[4]
                )

            # optimizing digaussian fit
            popt, pcov = curve_fit(
                dg,
                xdata=np.array([x1, y1]),
                ydata=z1,
                p0=[0., 0., 21.5, 4.87, 1e-3]
            )
            print(popt)

            # defining 2D MultiGaussian function
            # def mg(arg, *args):
            #     return multigaussian(
            #         arg, 
            #         K=np.array([[args[0], args[1]], [args[2], args[3]]]), 
            #         A=np.array([args[4], args[5]]), 
            #         C=args[6],
            #         C0=args[7]
            #     )
            
            # optimizing 2D MultiGaussian fit
            # popt, pcov = curve_fit(
            #     mg, 
            #     xdata=np.array([x1, y1]).T, 
            #     ydata=z1,
            #     p0=[.1, 0., 0., .1, 0., 0., 1., 0.]
            # )
            # print(popt)
            
            # polynomial optimization
            # deg_max, A = 5, []
            # for deg in range(deg_max + 1):
            #     for ii in range(deg + 1):
            #         A.append(x1**ii * y1**(deg-ii))
            # A = np.array(A).T
            # coeffA, r, rank, s = np.linalg.lstsq(A, z1)
            # print(coeffA, r, rank, s)
            
            # sine-cosine optimization
            # deg_max, B = 50, [np.ones(shape=x1.shape)]
            # for deg in range(1, deg_max + 1):
            #     B.append(
            #         np.sin(deg * x1 * np.pi / (np.max(x1) - np.min(x1))) * 
            #         np.sin(deg * y1 * np.pi / (np.max(y1) - np.min(y1)))
            #     )
            #     B.append(
            #         np.cos(deg * x1 * np.pi / (np.max(x1) - np.min(x1))) * 
            #         np.cos(deg * y1 * np.pi / (np.max(y1) - np.min(y1)))
            #     )
            #     B.append(
            #         np.sin(deg * x1 * np.pi / (np.max(x1) - np.min(x1))) * 
            #         np.cos(deg * y1 * np.pi / (np.max(y1) - np.min(y1)))
            #     )
            #     B.append(
            #         np.cos(deg * x1 * np.pi / (np.max(x1) - np.min(x1))) * 
            #         np.sin(deg * y1 * np.pi / (np.max(y1) - np.min(y1)))
            #     )
            # B = np.array(B).T
            # coeffB, r, rank, s = np.linalg.lstsq(B, z1)
            # print(coeffB, r, rank, s)

            # np.savetxt(os.path.join(save_d, '_'.join((os.path.basename(sd), fpath))), np.array([x1, y1, z1]))

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x1, y1, z1)  # Init
            ax.scatter(x1, y1, dg(np.array([x1, y1]), *popt))  # DiGauss
            # ax.scatter(x1, y1, mg(np.array([x1, y1]).T, *popt))  # 2D MultiGauss
            # ax.scatter(x1, y1, np.matmul(A, coeffA))  # Poly
            # ax.scatter(x1, y1, np.matmul(B, coeffB))  # Fourier
            plt.show()


