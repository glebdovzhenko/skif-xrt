import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    fp = r'/Users/glebdovzhenko/Downloads/Нагрузка на 1-й кристалл/exp_data/1/front_surface.txt'

    a = np.genfromtxt(fp, comments='%')
    a = a[::10]

    x, y, z, dx, dy, dz = a.T
    print(dz.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.quiver(x, y, z, dx, dy, dz, length=10., normalize=True)
    
    plt.show()
