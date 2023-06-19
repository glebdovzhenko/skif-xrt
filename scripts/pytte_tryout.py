import sys
import os.path

import numpy as np
import matplotlib.pyplot as plt

from pyTTE import TakagiTaupin, TTcrystal, TTscan, Quantity


if __name__ == '__main__':
    xtal = TTcrystal(crystal='Si', hkl=[1,1,1], thickness=Quantity(1,'mm'))
    scan = TTscan(constant = Quantity(5,'keV'), scan = Quantity(np.linspace(-50,150,150),'urad'), polarization = 'sigma')
    tt = TakagiTaupin(xtal, scan)
    scan_vector, reflectivity, transmission = tt.run()
    tt.plot()
    plt.show()
