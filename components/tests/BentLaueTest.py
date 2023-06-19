import unittest as ut
import xrt.backends.raycing.materials as rm
from xrt.backends.raycing.oes import BentLaueCylinder, LauePlate
import numpy as np
from components import BentLaueParaboloidWithBump


class TestPWB(ut.TestCase):
    def setUp(self):
        """
        Is called before every test method
        """
        self.mat = rm.CrystalSi(hkl=(1, 1, 1), geom='Laue reflection', t=2.)
        self.obj = BentLaueParaboloidWithBump(
            name='test',
            pitch=np.pi / 2.,
            roll=0.,
            yaw=0.,
            alpha=0.,
            material=(self.mat,),
            targetOpenCL='CPU',
        )
        self.obj.bump_pars = {
            'Cx': 0.,
            'Cy': 0.,
            'Sx': 1.,
            'Sy': 1., 
            'Axy': 0., 
        }
    
    def test_init(self):
        self.assertEqual(self.obj.Rx, np.inf)
        self.assertEqual(self.obj.Ry, np.inf)
        self.assertEqual(self.obj.R, np.inf)
        self.assertEqual(self.obj.r_for_refl, 'x')
    
    def test_radii(self):
        self.obj.r_for_refl = 'x'
        self.obj.Rx = 1.
        self.obj.Ry = 2.
        self.assertAlmostEqual(self.obj.Rx, self.obj.R)

        self.obj.r_for_refl = 'y'
        self.obj.Rx = 3.
        self.obj.Ry = 4.
        self.assertAlmostEqual(self.obj.Ry, self.obj.R)

    def test_surface(self):
        for _ in range(5):
            Rx, Ry = 20e3 * (np.random.rand() - .5), 20e3 * (np.random.rand() - .5)
 
            xs, ys = np.linspace(-.9 * Rx, .9 * Rx, 107), np.linspace(-.9 * Ry, .9 * Ry, 107)
            xs, ys = np.meshgrid(xs, ys)
            
            self.obj.Rx = np.inf
            self.obj.Ry = np.inf
            zs = np.zeros(shape=xs.shape)
            self.assertTrue(np.allclose(zs, self.obj.local_z(xs, ys)))

            self.obj.Rx = Rx
            self.obj.Ry = np.inf
            zs = Rx * (1. - np.sqrt(1 - (xs / Rx)**2))
            self.assertTrue(np.allclose(zs, self.obj.local_z(xs, ys)))

            self.obj.Rx = np.inf
            self.obj.Ry = Ry
            zs = Ry * (1. - np.sqrt(1 - (ys / Ry)**2))
            self.assertTrue(np.allclose(zs, self.obj.local_z(xs, ys)))

            self.obj.Rx = Rx
            self.obj.Ry = Ry
            zs = Rx * (1. - np.sqrt(1 - (xs / Rx)**2)) + Ry * (1. - np.sqrt(1 - (ys / Ry)**2))
            self.assertTrue(np.allclose(zs, self.obj.local_z(xs, ys)))

    def test_surface_against_xrt(self):
        obj2 = BentLaueCylinder(
            name='test',
            crossSection='circ',
            pitch=np.pi / 2.,
            roll=0.,
            yaw=0.,
            alpha=0.,
            material=(self.mat,),
            targetOpenCL='CPU',
        )

        for _ in range(5):
            Rx, Ry = 10e3 * np.random.rand(), 10e3 * np.random.rand()

            self.obj.Rx = np.inf
            self.obj.Ry = Ry
            obj2.R = Ry
            xs, ys = np.linspace(-.9 * Rx, .9 * Rx, 107), np.linspace(-.9 * Ry, .9 * Ry, 107)
            xs, ys = np.meshgrid(xs, ys)

            self.assertTrue(np.allclose(obj2.local_z(xs, ys), self.obj.local_z(xs, ys)))
        else:
            obj2 = LauePlate(
                name='test',
                pitch=np.pi / 2.,
                roll=0.,
                yaw=0.,
                alpha=0.,
                material=(self.mat,),
                targetOpenCL='CPU',
            )
            self.obj.Rx = np.inf
            self.obj.Ry = np.inf

            self.assertTrue(np.allclose(obj2.local_z(xs, ys), self.obj.local_z(xs, ys)))

    def test_norm_against_xrt(self):
        obj2 = BentLaueCylinder(
            name='test',
            crossSection='circ',
            pitch=np.pi / 2.,
            roll=0.,
            yaw=0.,
            alpha=0.,
            material=(self.mat,),
            targetOpenCL='CPU',
        )

        for _ in range(5):
            Rx, Ry = 10e3 * np.random.rand(), 10e3 * np.random.rand()

            self.obj.Rx = np.inf
            self.obj.Ry = Ry
            obj2.R = Ry
            xs, ys = np.linspace(-.9 * Rx, .9 * Rx, 107), np.linspace(-.9 * Ry, .9 * Ry, 107)
            xs, ys = np.meshgrid(xs, ys)
            _, _, _, a1, b1, c1 = self.obj.local_n(xs, ys)
            _, _, _, a2, b2, c2 = obj2.local_n(xs, ys)

            self.assertTrue(np.allclose(a1, a2))
            self.assertTrue(np.allclose(b1, b2))
            self.assertTrue(np.allclose(c1, c2))
        else:
            obj2 = LauePlate(
                name='test',
                pitch=np.pi / 2.,
                roll=0.,
                yaw=0.,
                alpha=0.,
                material=(self.mat,),
                targetOpenCL='CPU',
            )

            self.obj.Rx = np.inf
            self.obj.Ry = np.inf
            _, _, _, a1, b1, c1 = self.obj.local_n(xs, ys)
            _, _, _, a2, b2, c2 = obj2.local_n(xs, ys)

            self.assertTrue(np.allclose(a1, a2))
            self.assertTrue(np.allclose(b1, b2))
            self.assertTrue(np.allclose(c1, c2))

    def test_bnorm_blc(self):
        obj2 = BentLaueCylinder(
            name='test',
            crossSection='circ',
            pitch=np.pi / 2.,
            roll=0.,
            yaw=0.,
            alpha=0.,
            material=(self.mat,),
            targetOpenCL='CPU',
        )

        for _ in range(5):
            Rx, Ry = 5. * np.random.rand(), 5. * np.random.rand()
            alpha = (np.random.rand() - .5) * np.pi

            self.obj.Rx = np.inf
            self.obj.Ry = Ry
            obj2.R = Ry
            self.obj.set_alpha(alpha)
            obj2.set_alpha(alpha)
            xs, ys = np.linspace(-.9 * Rx, .9 * Rx, 107), np.linspace(-.9 * Ry, .9 * Ry, 107)
            xs, ys = np.meshgrid(xs, ys)
            a1, b1, c1, _, _, _ = self.obj.local_n(xs, ys)
            a2, b2, c2, _, _, _ = obj2.local_n(xs, ys)

            self.assertTrue(np.allclose(a1, a2))
            self.assertTrue(np.allclose(b1, b2))
            self.assertTrue(np.allclose(c1, c2))

    def test_bnorm_lp(self):
        obj2 = LauePlate(
            name='test',
            pitch=np.pi / 2.,
            roll=0.,
            yaw=0.,
            alpha=0.,
            material=(self.mat,),
            targetOpenCL='CPU',
        )

        for _ in range(5):
            Rx, Ry = 5. * np.random.rand(), 5. * np.random.rand()
            alpha = (np.random.rand() - .5) * np.pi

            self.obj.Rx = np.inf
            self.obj.Ry = np.inf
            self.obj.set_alpha(alpha)
            obj2.set_alpha(alpha)
            xs, ys = np.linspace(-.9 * Rx, .9 * Rx, 107), np.linspace(-.9 * Ry, .9 * Ry, 107)
            xs, ys = np.meshgrid(xs, ys)
            a1, b1, c1, _, _, _ = self.obj.local_n(xs, ys)
            a2, b2, c2, _, _, _ = obj2.local_n(xs, ys)
            
            self.assertTrue(np.allclose(a1, a2))
            self.assertTrue(np.allclose(b1, b2))
            self.assertTrue(np.allclose(c1, c2))


if __name__ == '__main__':
    ut.main()
