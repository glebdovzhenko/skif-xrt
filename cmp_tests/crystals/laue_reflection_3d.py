import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from utils.pgutils import apply_cmap
from scipy.signal import resample


if __name__ == '__main__':
    data = np.load('result.npy')

    size_x = data.shape[0]
    size_y = data.shape[1]
    size_z = data.shape[2]

    re_size_x = 500
    re_size_y = 500
    re_size_z = size_z

    print(data.shape)

    data = data[
        data.shape[0] // 2 - size_x // 2:data.shape[0] // 2, #+ size_x // 2,
        data.shape[1] // 2 - size_y // 2:data.shape[1] // 2, #+ size_y // 2,
        data.shape[2] // 2 - size_z // 2:data.shape[2] // 2 + size_z // 2
    ]

    print(data.shape)

    if re_size_x != size_x:
        data = np.apply_along_axis(lambda x: resample(x, re_size_x), 0, data)
    if re_size_y != size_y:
        data = np.apply_along_axis(lambda x: resample(x, re_size_y), 1, data)
    if re_size_z != size_z:
        data = np.apply_along_axis(lambda x: resample(x, re_size_z), 2, data)

    print(data.shape)

    data_rgb = apply_cmap(data)
    data_rgb[..., 3] = 0.2
    data_rgb = ((data_rgb * 254) % 255).astype(int)

    print(data_rgb.shape)

    app = pg.mkQApp("GLVolumeItem Example")
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle('pyqtgraph example: GLVolumeItem')
    w.setCameraPosition(distance=200)

    v = gl.GLVolumeItem(data_rgb)
    v.translate(-re_size_x//2, -re_size_y//2, -re_size_z//2)
    w.addItem(v)

    ax = gl.GLAxisItem()
    w.addItem(ax)

    pg.exec()
