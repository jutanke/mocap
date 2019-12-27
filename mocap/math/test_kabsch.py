import sys
sys.path.insert(0, './../..')
import unittest
import numpy as np
import numpy.random as rnd
import numpy.linalg as la
import mocap.math.geometry as gm
from mocap.math.kabsch import kabsch
from math import pi  # pylint: disable=no-name-in-module


class TestKabsch(unittest.TestCase):

    def test_norotation(self):
        n_joints = 16
        Q = rnd.rand(n_joints, 3)
        P = Q.copy() + np.array([[0, 0, 5]])
        R = kabsch(P, Q)
        self.assertAlmostEqual(la.det(R), 1.0, places=4)
        self.assertAlmostEqual(np.sum(R), 3.0, places=4)
        self.assertAlmostEqual(R[0, 0], 1.0, places=4)
        self.assertAlmostEqual(R[1, 1], 1.0, places=4)
        self.assertAlmostEqual(R[2, 2], 1.0, places=4)

    def test_rotation(self):
        n_joints = 16
        for i in range(25):
            Q = rnd.rand(n_joints, 3)
            P = Q.copy()
            a, b, c = rnd.rand(3) * pi * 2
            R = gm.rot3d(a, b, c)
            P = P @ np.transpose(R)
            P += np.array([[0, 0, i]])

            R_pred = kabsch(Q, P)

            dist = np.sum(np.abs(R_pred - R))
            self.assertAlmostEqual(dist, 0.0, places=4)




if __name__ == '__main__':
    unittest.main()
