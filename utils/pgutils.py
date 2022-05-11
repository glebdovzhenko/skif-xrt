import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Qt.QtGui as QtGui
from pyqtgraph.Qt.QtCore import Qt
import pickle
import os
import numpy as np


def apply_cmap(vals, log_scale=False):

    red_cheb = pickle.loads(
        b'\x80\x04\x95\xa0\x01\x00\x00\x00\x00\x00\x00\x8c\x1anumpy.polynomial.chebyshev\x94\x8c\tChebyshev\x94\x93'
        b'\x94)\x81\x94}\x94(\x8c\x04coef\x94\x8c\x15numpy.core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c'
        b'\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x0e\x85\x94h\t\x8c\x05'
        b'dtype\x94\x93\x94\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff'
        b'\xffK\x00t\x94b\x89Cp\x87\xa8\x1d\xc82*\xe0?As\xc9\xd6\x93\xca\xd3?\xe4\xd3\xbd1H\xda\xbe\xbfI\xb7\x99tB'
        b'\xcc\xcd\xbf\xbe\xf8\xdepp\x08\xa8\xbf\x89\xd1G\x98\x1c\xac\xb8?\xea\xcd\xcb\xdfb\xc5\x90\xbf\x9f\xc1='
        b'\xe3Uh\xa2\xbfrU\xadA\xc3t\x98?\xc7\x88\xc1\x1a=0a?\xdd-\xfc\xb1P\x8a\x8e\xbfF>\xf0\x88\x16k\x80?{\xc8\x93'
        b'\x98\x8c\xf6r?\x1e\x84\xb83\xb7H\x83\xbf\x94t\x94b\x8c\x06domain\x94h\x08h\x0bK\x00\x85\x94h\r\x87\x94R'
        b'\x94(K\x01K\x02\x85\x94h\x15\x89C\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0o@\x94t\x94b'
        b'\x8c\x06window\x94h\x08h\x0bK\x00\x85\x94h\r\x87\x94R\x94(K\x01K\x02\x85\x94h\x15\x89C\x10\x00\x00\x00\x00'
        b'\x00\x00\xf0\xbf\x00\x00\x00\x00\x00\x00\xf0?\x94t\x94bub.'
    )

    green_cheb = pickle.loads(
        b'\x80\x04\x95\xa0\x01\x00\x00\x00\x00\x00\x00\x8c\x1anumpy.polynomial.chebyshev\x94\x8c\tChebyshev\x94\x93'
        b'\x94)\x81\x94}\x94(\x8c\x04coef\x94\x8c\x15numpy.core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c'
        b'\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x0e\x85\x94h\t\x8c\x05'
        b'dtype\x94\x93\x94\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff'
        b'\xffK\x00t\x94b\x89Cp\x8a\xa2\xd3/2\xfa\xdb?3\xe4\t\xa9\x00\xd0\xb8\xbf\xf1\xa6w\xb7\x9a\x84\xdd\xbf\xc3R'
        b'\xdcx\xa7Y\xb1?\xe0\xe8A\xfa\x12\xca\xb4?\x06;\xea\xa0\xd7\xb5o?\x04\xea\t\x9c\x8eh\x87\xbf\x179\xaa\x00+'
        b'\xcej\xbfO\xad2=m\xe5e\xbf\x1c\xc61"\xa9\x1a"?\xe4\xb0diT7g?\x90\xf4\xa17\x03\xa0Q?\xab\xcfC\xb0vYc?f\x90/'
        b'\xbd\x1bb@\xbf\x94t\x94b\x8c\x06domain\x94h\x08h\x0bK\x00\x85\x94h\r\x87\x94R\x94(K\x01K\x02\x85\x94h\x15'
        b'\x89C\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0o@\x94t\x94b\x8c\x06window\x94h\x08h\x0bK'
        b'\x00\x85\x94h\r\x87\x94R\x94(K\x01K\x02\x85\x94h\x15\x89C\x10\x00\x00\x00\x00\x00\x00\xf0\xbf\x00\x00\x00'
        b'\x00\x00\x00\xf0?\x94t\x94bub.'
    )

    blue_cheb = pickle.loads(
        b'\x80\x04\x95\xa0\x01\x00\x00\x00\x00\x00\x00\x8c\x1anumpy.polynomial.chebyshev\x94\x8c\tChebyshev\x94\x93'
        b'\x94)\x81\x94}\x94(\x8c\x04coef\x94\x8c\x15numpy.core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c'
        b'\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x0e\x85\x94h\t\x8c\x05'
        b'dtype\x94\x93\x94\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff'
        b'\xffK\x00t\x94b\x89Cp=M\x95\x8f\x86\xa6\xd6?"r`a\xd1\xfd\xd6\xbf~7\xf3\xfe\xcf\x11\xbf\xbfu\x81"\r\xcc\xe2'
        b'\xcc?\xa4\xe97\xe8\xe9\xc8\xc2\xbf;D\x08\x86\x98/\xa5?ZRN\r\tB\xa7?\xf7\x10\xff\'\x81\xb6\x97\xbf\xd9\xfd'
        b'\xa7\x92W\xb4\x93\xbfA\xc3\xf4ukvR?F\xa0\xaf\x18\x82\xcf\x82?C\tO.0\x1f{?|\x1c\xae,=\xf9\x85\xbf\xb6f\x04/P'
        b'\xadr\xbf\x94t\x94b\x8c\x06domain\x94h\x08h\x0bK\x00\x85\x94h\r\x87\x94R\x94(K\x01K\x02\x85\x94h\x15\x89C'
        b'\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0o@\x94t\x94b\x8c\x06window\x94h\x08h\x0bK\x00'
        b'\x85\x94h\r\x87\x94R\x94(K\x01K\x02\x85\x94h\x15\x89C\x10\x00\x00\x00\x00\x00\x00\xf0\xbf\x00\x00\x00\x00'
        b'\x00\x00\xf0?\x94t\x94bub.'
    )

    colors = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))

    if log_scale:
        colors = np.log(1. + colors)
        colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))
    colors *= 255

    return np.array([
        red_cheb(colors), green_cheb(colors), blue_cheb(colors), np.ones_like(colors)
    ]).transpose(tuple(range(1, len(colors.shape) + 1)) + (0, ))


class PgSurfacePlot(gl.GLViewWidget):
    def __init__(self, parent=None, devicePixelRatio=None, rotationMethod='euler',
                 box=None):
        gl.GLViewWidget.__init__(
            self,
            parent=parent,
            devicePixelRatio=devicePixelRatio,
            rotationMethod=rotationMethod
        )

        self.initUI()

    def initUI(self):
        self.text = "Лев Николаевич Толстой\nАнна Каренина"
        self.show()

    def paintEvent(self, event):
        gl.GLViewWidget.paintEvent(self, event)
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawText(event, qp)
        qp.end()

    def drawText(self, event, qp):
        qp.setPen(QtGui.QColor(255, 255, 255))
        qp.setFont(QtGui.QFont('Decorative', 10))
        coord = QtGui.QVector4D(0, 0, 0, 1)
        coord = self.viewMatrix().map(coord)
        coord = self.projectionMatrix().map(coord)
        qp.drawText(int(coord.x()), int(coord.y()), self.text)


def surf3d(xdata, ydata, zdata, title=""):
    def scale(vec, nmin, nmax, dmin=None, dmax=None):
        if dmin is None:
            dmin = np.min(vec)
        if dmax is None:
            dmax = np.max(vec)
        return nmin + (nmax - nmin) * (vec - dmin) / (dmax - dmin)

    app = pg.mkQApp(title)
    box = ((-5, 5), (-5, 5), (-5, 5))
    w = PgSurfacePlot()
    w.show()
    w.setWindowTitle(title)
    w.setCameraPosition(distance=100)

    xgrid = gl.GLGridItem()
    ygrid = gl.GLGridItem()
    zgrid = gl.GLGridItem()
    w.addItem(xgrid)
    w.addItem(ygrid)
    w.addItem(zgrid)

    xgrid.scale(0.05 * (box[2][1] - box[2][0]), 0.05 * (box[1][1] - box[1][0]), 1.)
    ygrid.scale(0.05 * (box[0][1] - box[0][0]), 0.05 * (box[2][1] - box[2][0]), 1.)
    zgrid.scale(0.05 * (box[0][1] - box[0][0]), 0.05 * (box[1][1] - box[1][0]), 1.)

    xgrid.rotate(90, 0, 1, 0)
    ygrid.rotate(90, 1, 0, 0)

    xgrid.translate(box[0][0], 0.5 * (box[1][1] + box[1][0]), 0.5 * (box[2][1] + box[2][0]))
    ygrid.translate(0.5 * (box[0][1] + box[0][0]), box[1][0], 0.5 * (box[2][1] + box[2][0]))
    zgrid.translate(0.5 * (box[0][1] + box[0][0]), 0.5 * (box[1][1] + box[1][0]), box[2][0])

    p2 = gl.GLSurfacePlotItem(
        x=scale(xdata, *box[0]),
        y=scale(ydata, *box[1]),
        z=scale(zdata, *box[2]),
        colors=apply_cmap(zdata),
        shader=None,
        smooth=True,
    )
    w.addItem(p2)

    pg.exec()


if __name__ == '__main__':
    xs = np.linspace(0., np.pi, 100)
    ys = np.linspace(0., np.pi, 100)
    zs = np.sin(xs.reshape((-1, 1)) ** 2 + ys.reshape((1, -1)) ** 2)
    surf3d(xs, ys, zs)
