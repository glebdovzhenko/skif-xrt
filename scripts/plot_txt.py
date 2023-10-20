import numpy as np
from matplotlib import pyplot as plt
import os

if __name__ == '__main__':
    d1 = np.genfromtxt(os.path.join(
            '/Users', 'glebdovzhenko', 'Downloads', 
            'Telegram Desktop', 'f15fmDiamond25SiC05.txt'
        ),
        skip_header=1,
    )
    print(d1.T)

    d2 = np.genfromtxt(os.path.join(
            '/Users', 'glebdovzhenko', 'Yandex.Disk.localized', 'Dev', 
            'skif-xrt', 'datasets', 'nstu-scw-2', 'absorbed_power', 'C00-XZ.txt'
        ),
        skip_header=1,
    )
    print(d2.T)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(*d1.T)
    ax.scatter(*d2.T)
    plt.show()

