import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit
from components.bump_eqs import crs_xy
import os
import pandas as pd


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


if __name__ == '__main__':
    wd = r'/Users/glebdovzhenko/Downloads/Нагрузка на 1-й кристалл/exp_data'
    save_path = os.path.join(os.getenv('BASE_DIR'), 'datasets', 'bump.csv')
    dataset = pd.DataFrame(columns=['pt', 'surface', 'Cx', 'Cy', 'Sx', 'Sy', 'Axy', 'Rx', 'Ry'])

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
                return crs_xy(
                    arg[0], arg[1],
                    Cx=args[0],
                    Cy=args[1],
                    Sx=args[2],
                    Sy=args[3],
                    Rx=0.,
                    Ry=0.,
                    Axy=args[4]
                )

            # optimizing digaussian fit
            popt, pcov = curve_fit(
                dg,
                xdata=np.array([x1, y1]),
                ydata=z1,
                p0=[0., 0., 21.5, 4.87, 1e-3]
            )
            print(popt)
            dataset.loc[dataset.shape[0]] = {
                'Cx': popt[0],
                'Cy': popt[1],
                'Sx': popt[2],
                'Sy': popt[3],
                'Rx': 0.,
                'Ry': 0.,
                'Axy': popt[4],
                'pt': int(os.path.basename(sd)),
                'surface': fpath.replace('.txt', '').replace('_surface', '')
            }
            
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x1, y1, z1)
            ax.scatter(x1, y1, dg(np.array([x1, y1]), *popt)) 
            plt.show()
    
    dataset.to_csv(save_path)
