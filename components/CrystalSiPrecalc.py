import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.myopencl as mcl
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import os
from functools import reduce
from scipy.interpolate import griddata
from matplotlib import pyplot as plt


class CrystalSiPrecalc(rm.CrystalSi):
    """
    Param columns:
    'en': energy in eV
    't': crystal thickness in mcm
    'r': bending radius in mm
    'chi': asymmetry angle in degrees
    """
    db_columns = ['en', 't', 'r', 'chi', 'fit_a', 'fit_c', 'fit_hw', 'fit_gw', 'fit_lw', 'fit_pv', 'fit_skew']

    def __init__(self, *args, **kwargs):
        if 'database' not in kwargs.keys():
            self.db_addr = '/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/components/Si111ref.csv'
        else:
            self.db_addr = kwargs.pop('database')

        rm.CrystalSi.__init__(self, *args, **kwargs)
        
        self.db = None
        self.init_db()

        self.thmin = -2000 * 1e-6
        self.thmax = 2000 * 1e-6

    @staticmethod
    def params_to_db_values(**kwargs):
        result = kwargs.copy()
        result['en'] = np.around(result['en']).astype(int)
        result['t'] = np.around(1e3 * result['t']).astype(int)
        result['r'] = result['r'] if np.isinf(result['r']) else np.around(result['r']).astype(int)
        result['chi'] = np.around(np.degrees(result['chi'])).astype(int)
        return result

    @staticmethod
    def db_values_to_params(**kwargs):
        result = kwargs.copy()
        result['en'] = float(result['en'])
        result['t'] = 1e-3 * result['t']
        result['r'] = float(result['r'])
        result['chi'] = np.radians(float(result['chi']))
        return result

    def init_db(self):
        if os.path.exists(self.db_addr):
            self.db = pd.read_csv(self.db_addr, index_col=0)
        else:
            self.db = pd.DataFrame(columns=self.db_columns)

    def save_db(self):
        self.db.to_csv(self.db_addr)

    def db_locate(self, **kwargs):
        db_pars = self.params_to_db_values(**kwargs)
        print("##### LOCATING PARAMS: " + ", ".join("%s = %f" % (k, v) for (k, v) in db_pars.items()))

        params = self.db.loc[
            reduce(lambda a, b: a & b, map(lambda c: self.db[c[0]] == c[1], db_pars.items()))
        ].squeeze()
        if params.shape[0] == 0:
            print("##### PARAMS NOT FOUND")
            return None

        print("##### FOUND PARAMS: " +
              ", ".join("%s = %f" % (k, params[k]) for k in 
                        ('fit_c', 'fit_hw', 'fit_gw', 'fit_lw', 'fit_pv', 'fit_a', 'fit_skew')))
        return params.to_dict()

    def db_interpolate(self, **kwargs):
        xs = self.params_to_db_values(**kwargs)
        inf_r = np.isinf(xs['r'])

        if inf_r:
            db_xs_fields = 't', 'chi'
        else:
            db_xs_fields = 't', 'r', 'chi'

        if isinstance(xs['en'], np.ndarray):
            # xs = np.array([xs['en'], [xs['t']] * xs['en'].size,
            #                [xs['r']] * xs['en'].size, [xs['chi']] * xs['en'].size]).T
            xs = np.array([xs['en'], *[[xs[f]] * xs['en'].size for f in db_xs_fields]]).T
        else:
            # xs = np.array([[xs['en'], xs['t'], xs['r'], xs['chi']]])
            xs = np.array([[xs['en']] + [xs[f] for f in db_xs_fields]])
        
        if inf_r:
            db_xs = self.db.loc[np.isinf(self.db['r'])]
        else:
            db_xs = self.db.loc[~np.isinf(self.db['r'])]

        
        db_ys = db_xs.loc[:, [col for col in self.db_columns if 'fit_' in col]]
        db_xs = db_xs.loc[:, ['en', *db_xs_fields]].to_numpy()

        # if not inf_r:
        #     ii = ~np.isinf(db_xs[:, 2])
        #     db_xs = db_xs[ii]
        
        result = dict()
        for par in ['fit_c', 'fit_hw', 'fit_gw', 'fit_lw', 'fit_pv', 'fit_a', 'fit_skew']:
            result[par] = griddata(db_xs, db_ys.loc[:, par].to_numpy(), xs, method='linear').squeeze()[()]
        return result

    def db_add(self, **kwargs):
        """
        Calculates a reflectivity curve using CrystalSi.get_amplitude_TT() and adds it to the database.
        """
        db_pars = self.params_to_db_values(**kwargs)
        calc_pars = self.db_values_to_params(**db_pars)
        print("##### ADDING PARAMS: " + ", ".join("%s = %f" % (k, v) for (k, v) in db_pars.items()))

        params = self.db.loc[
            reduce(lambda a, b: a & b, map(lambda c: self.db[c[0]] == c[1], db_pars.items()))
        ].squeeze()
        if params.shape[0] != 0:
            print("##### PARAMS ALREADY THERE")
            return

        thetas = np.linspace(self.thmin, self.thmax, 10000)  # rad
        p1 = self.calc_params(thetas, **calc_pars)
        thetas_hw = max(np.abs(p1[1]) + 3. * np.abs(p1[2]), 5e-5)
        thetas = np.linspace(p1[0] - thetas_hw, p1[0] + thetas_hw, 10000)
        p = self.calc_params(thetas, **calc_pars)
        
        res = {
            **db_pars,
            'fit_c': p[0], 'fit_hw': p[1], 'fit_gw': p[2], 'fit_lw': p[3], 'fit_pv': p[4], 
            'fit_a': p[5], 'fit_skew': p[6]
        }
        self.db.loc[int(np.nanmax([self.db.shape[0], self.db.index.max() + 1]))] = res
        print("##### ADDED PARAMS: " + ", ".join("%s = %f" % (k, v) for (k, v) in res.items()))
    
    def db_add_ext(self, **kwargs):
        """
        Adds to the database a reflectivity curve calculated externally.
        """
        thetas = kwargs.pop('thetas')
        ref = kwargs.pop('ref')
        
        db_pars = self.params_to_db_values(**kwargs)
        print("##### ADDING PARAMS: " + ", ".join("%s = %f" % (k, v) for (k, v) in db_pars.items()))

        params = self.db.loc[
            reduce(lambda a, b: a & b, map(lambda c: self.db[c[0]] == c[1], db_pars.items()))
        ].squeeze()
        if params.shape[0] != 0:
            print("##### PARAMS ALREADY THERE")
            return

        p = self.calc_params(thetas=thetas, ref=ref)
        res = {
            **db_pars,
            'fit_c': p[0], 'fit_hw': p[1], 'fit_gw': p[2], 'fit_lw': p[3], 'fit_pv': p[4], 
            'fit_a': p[5], 'fit_skew': p[6]

        }
        self.db.loc[int(np.nanmax([self.db.shape[0], self.db.index.max() + 1]))] = res
        print("##### ADDED PARAMS: " + ", ".join("%s = %f" % (k, v) for (k, v) in res.items()))


    def calc_reflectivity(self, thetas, **kwargs):
        plane_norm = (0., np.sin(kwargs['chi']), np.cos(kwargs['chi']))
        surface_norm = (0., -1., 0.)

        theta0 = np.arcsin(rm.ch / (2 * self.d * kwargs['en']))
        k_inc = (
            np.zeros_like(thetas), np.cos(thetas + theta0 + kwargs['chi']), -np.sin(thetas + theta0 + kwargs['chi']))
        k_ref = (
            np.zeros_like(thetas), np.cos(thetas + theta0 - kwargs['chi']), np.sin(thetas + theta0 - kwargs['chi']))

        kwgs = {
            'E': kwargs['en'] * np.ones_like(thetas),
            'beamInDotNormal': np.dot(surface_norm, k_inc),
            'beamOutDotNormal': np.dot(surface_norm, k_ref),
            'beamInDotHNormal': np.dot(plane_norm, k_inc),
            'ucl': mcl.XRT_CL(r'materials.cl', targetOpenCL='CPU'),
            'alphaAsym': kwargs['chi'],
            'Rcurvmm': kwargs['r']
        }

        self.t = kwargs['t']
        return rm.CrystalSi.get_amplitude_TT(self, **kwgs)

    def calc_params(self, thetas, ref=None, **kwargs):
        
        if ref is None:
            c_s, c_p = self.calc_reflectivity(thetas, **kwargs)
            ref = np.sqrt(.5 * np.abs(c_s) ** 2 + .5 * np.abs(c_p) ** 2)

        br = np.sum((.5 * ref[1:] + .5 * ref[:-1]) * (thetas[1:] - thetas[:-1])) / np.max(ref)
        p0 = [np.sum(thetas * ref) / np.sum(ref), .5 * br, .5 * br, .5 * br, .5, np.max(ref), 0.]
        mit, mat = np.min(thetas), np.max(thetas)

        # plt.plot(thetas, np.abs(ref), label=r'$R_{TT}$')
        # plt.plot(thetas, self.f(thetas, *p0), label=r'Fit')
        # plt.legend()
        # plt.show()

        p, _ = curve_fit(
            self.f, thetas, ref, p0, maxfev=10000, 
            bounds=((mit, 0.,        0.,        0.,        0., 0., -np.inf), 
                    (mat, mat - mit, mat - mit, mat - mit, 1., 2.,  np.inf))
        )

        # plt.plot(thetas, np.abs(ref), label=r'$R_{TT}$')
        # plt.plot(thetas, self.f(thetas, *p), label=r'Fit')
        # plt.legend()
        # plt.show()

        return p

    @staticmethod
    def f(x, *args):
        """
        Gaussian: FWHM = 2. * np.sqrt(2. * np.log(2)) * sigma
        Lorentzian: FWHM = gamma
        :param x:
        :param args: [0] center, [1] heaviside width, [2] gaussian FWHM, [3] Lorentzian FWHM, [4] p-Voigt eta
        [5] amplitude, [6] skew
        :return:
        """
        result = np.zeros(shape=x.shape)

        gs = args[2] / 2. * np.sqrt(2. * np.log(2))
        lg = args[3]
        rw = (x - args[0]) > np.abs(.5 * args[1])
        lw = (x - args[0]) < -np.abs(.5 * args[1])
        cc = np.abs(x - args[0]) < np.abs(.5 * args[1])

        result[cc] = 1.

        result[rw] = args[4] * np.exp(-((x[rw] - args[0] - .5 * np.abs(args[1])) / gs) ** 2 / 2.)
        result[lw] = args[4] * np.exp(-((x[lw] - args[0] + .5 * np.abs(args[1])) / gs) ** 2 / 2.)
        result[rw] += (1. - args[4]) * lg ** 2. / ((x[rw] - args[0] - .5 * np.abs(args[1]))**2 + lg ** 2)
        result[lw] += (1. - args[4]) * lg ** 2. / ((x[lw] - args[0] + .5 * np.abs(args[1]))**2 + lg ** 2)

        result *= args[5]
        result *= args[6] * (x - args[0]) + 1.

        return result

    def get_amplitude_TT(self, E, beamInDotNormal, beamOutDotNormal=None, beamInDotHNormal=None, alphaAsym=None,
                         Rcurvmm=None, ucl=None):
        """
        Overrides CrystalSi.get_amplitude_TT with database lookup.
        """
        params = self.db_locate(en=np.mean(E), t=self.t, r=Rcurvmm, chi=alphaAsym)
        
        if params is None:
            params = self.db_interpolate(en=np.mean(E), t=self.t, r=Rcurvmm, chi=alphaAsym)

        if any(map(lambda x: x is None or np.isnan(x), params.values())):
            raise ValueError('##### COULD NOT INTERPOLATE')

        thetaB = self.get_Bragg_angle(E)
        if alphaAsym > 0:
            thetas = np.arccos(-beamInDotNormal) - alphaAsym - thetaB
        else:
            thetas = -(np.arccos(-beamInDotNormal) + alphaAsym + thetaB)

        c_s = self.f(
            thetas, params['fit_c'], params['fit_hw'], params['fit_gw'], params['fit_lw'], params['fit_pv'],
            params['fit_a'], params['fit_skew']
        )
        c_p = self.f(
            thetas, params['fit_c'], params['fit_hw'], params['fit_gw'], params['fit_lw'], params['fit_pv'],
            params['fit_a'], params['fit_skew']
        )

        return c_s, c_p


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    cr = CrystalSiPrecalc(geom='Laue reflected', hkl=(1, 1, 1), t=1., factDW=1., useTT=True)

    # ############################################### ADDING INFO TO DB ################################################

    # cr.thmin = -5000 * 1e-6
    # cr.thmax = 5000 * 1e-6

    en = 29.5e3  # eV
    alpha = np.radians(37.)
    crR = 2000.0  # mm
    crT = 1.7 # mm
    
    for crT in [1.6, 1.8, 2.0, 2.2, 2.4]:
        cr.t = crT
        for alpha in [np.radians(35.), np.radians(36.), np.radians(-35.), np.radians(-36.)]:
            for crR in [18.e3, 20.e3, 22.e3, np.inf]:
                cr.db_add(en=en, t=crT, r=crR, chi=alpha)
    
    cr.save_db()

    # ################################################### PLOTTING #####################################################

    # en = 25.e3  # eV
    # alpha = np.radians(20.)
    # crR = 2000.0  # mm
    # crT = 2.  # mm
    #
    # for en in [25e3, 30e3, 35e3, 40e3]:
    #     params = cr.db_locate(en=en, t=crT, r=crR, chi=alpha)
    #     if params is None:
    #         raise ValueError('Parameter set not found in DB')
    #
    #     thetas = np.linspace(-1000, 500, 500) * 1e-6
    #
    #     plt.plot(thetas, cr.f(thetas, params['fit_c'], params['fit_hw'], params['fit_gw'], params['fit_a'],
    #                           params['fit_skew']), label='%d keV' % np.round(en * 1e-3))
    # plt.legend()
    # plt.show()

    # ################################################ INTERPOLATION ###################################################

    en = 30.e3  # eV
    alpha = np.radians(35.3)
    crR = 20000.  # mm
    crT = 2.  # mm
    params = cr.db_interpolate(en=np.array([30.1e3, 29.9e3]), t=crT, r=crR, chi=alpha)
    params = pd.DataFrame(params)
    thetas = np.linspace(-1000, 500, 500) * 1e-6
    print(params) 
    for ii in params.index:
        pr = params.loc[ii].to_dict()
        plt.plot(thetas, cr.f(thetas, pr['fit_c'], pr['fit_hw'], pr['fit_gw'], pr['fit_lw'], pr['fit_pv'], 
                              pr['fit_a'], pr['fit_skew']), '--')

    # for en in [25e3, 30e3]:
    #     params = cr.db_locate(en=en, t=crT, r=crR, chi=alpha)
    #     if params is None:
    #         raise ValueError('Parameter set not found in DB')
    
    #     plt.plot(thetas, cr.f(thetas, params['fit_c'], params['fit_hw'], params['fit_gw'], params['fit_lw'], 
    #                           params['fit_pv'], params['fit_a'], params['fit_skew']), 
    #              label='%d keV' % np.round(en * 1e-3))
    plt.show()