import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from components import CrystalSiPrecalc 


if __name__ == '__main__':
    # For Si meridional radius is ~6 times larger than sagittal
    d = np.loadtxt('/Users/glebdovzhenko/Dropbox/Documents/07_SKIF/09_oasys/SKIF-1-5/wd/diff_pat.dat',
                   skiprows=5)
    thetas = d[:, 0] * 1e-6
    ref = d[:, -1]

    cr = CrystalSiPrecalc(geom='Laue reflected', hkl=(1, 1, 1), t=1., factDW=1., useTT=True, 
                          database='/Users/glebdovzhenko/Dropbox/PycharmProjects/skif-xrt/components/Si111ref_sag.csv')
    
    en = 35.e3  # eV
    alpha = np.radians(-30.)
    crR = 100000.0  # mm
    crT = .2  # mm

    cr.t = crT
    
    # cr.db_add_ext(en=en, t=crT, r=crR, chi=alpha, thetas=thetas, ref=ref)
    # cr.save_db()
    
    for crR in np.linspace(-100., 100., 20) * 1e3:
        en = (25. + 10. * np.random.rand(7)) * 1e3
        params = cr.db_interpolate(en=en, t=crT, r=crR, chi=alpha)
        params = pd.DataFrame(params)
        thetas = np.linspace(-1000, 500, 500) * 1e-6
        
        plt.figure()
        plt.title('R = %.01f m' % (crR * 1e-3))
        for ii in params.index:
            pr = params.loc[ii].to_dict()
            plt.plot(thetas, cr.f(thetas, pr['fit_c'], pr['fit_hw'], pr['fit_gw'], pr['fit_a'],
                                  pr['fit_skew']), label='E = %.01f keV' % (en[ii] * 1e-3))
        plt.legend()
        plt.tight_layout()
        plt.show()
    # for en in [25e3, 30e3]:
        # params = cr.db_locate(en=en, t=crT, r=crR, chi=alpha)
        # if params is None:
            # raise ValueError('Parameter set not found in DB')
    
        # plt.plot(thetas, cr.f(thetas, params['fit_c'], params['fit_hw'], params['fit_gw'], params['fit_a'],
                              # params['fit_skew']), label='%d keV' % np.round(en * 1e-3))
    # plt.show()
