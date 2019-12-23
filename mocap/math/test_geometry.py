import sys
sys.path.insert(0, './../..')
import unittest
import numpy as np
import numpy.linalg as la
import mocap.math.geometry as gm
from math import pi  # pylint: disable=no-name-in-module


class Test3DRotation(unittest.TestCase):

    def test_apply_3drot(self):
        person = np.array([
            (5, 0, 0),
            (4, 0, 0),
            (3, 0, 0),
            (2, 0, 0),
            (1, 0, 0)
        ], np.float32)

        gt_person_rot = np.array([
            (3,   2, 0),
            (3,   1, 0),
            (3,   0, 0),
            (3,  -1, 0),
            (3,  -2, 0)
        ], np.float32)

        R = gm.rot3d(0, 0, pi/2)
        person_rot = gm.apply_rotation(person, R)

        diff = np.sum(person_rot - gt_person_rot)
        self.assertAlmostEqual(0, diff)

    def test_3drot(self):
        a = np.array([1, 0, 0])
        R = gm.rot3d(0, pi/2, 0)
        b = R @ a

        self.assertAlmostEqual(b[0], 0)
        self.assertAlmostEqual(b[1], 0)
        self.assertAlmostEqual(b[2], -1)

    def test_find_3d_normal_on_plane(self):
        left = np.array([-1, 0, 0], np.float32)
        right = np.array([1, 0, 0], np.float32)
        plane = np.array([0, 1], np.int32)

        n = gm.get_3d_normal_on_plane(left, right, plane)

        self.assertAlmostEqual(n[0], 0)
        self.assertAlmostEqual(n[1], 1)
        self.assertAlmostEqual(n[2], 0)


class TestDistances(unittest.TestCase):

    def test_distances(self):
        A = np.array([
            (1, 1),
            (2, 2),
            (3, 3)
        ], np.float32)
        B = np.array([
            (1, 5),
            (2, 4),
            (3, 3)
        ], np.float32)

        d = gm.distances(A, B)
        self.assertEqual(len(d), 3)
        self.assertAlmostEqual(d[0], 4)
        self.assertAlmostEqual(d[1], 2)
        self.assertAlmostEqual(d[2], 0)


class Test2DRotation(unittest.TestCase):

    def test_simple2d_rotation(self):
        alpha = 0
        R = gm.rot2d(alpha)
        self.assertEqual(len(R.shape), 2)
        self.assertEqual(R.shape[0], 2)
        self.assertEqual(R.shape[1], 2)

        self.assertAlmostEqual(R[0,0], 1)
        self.assertAlmostEqual(R[1,1], 1)
        self.assertAlmostEqual(R[1,0], 0)
        self.assertAlmostEqual(R[0,1], 0)

    def test_rot90(self):
        alpha = pi/2
        R = gm.rot2d(alpha)
        self.assertEqual(len(R.shape), 2)
        self.assertEqual(R.shape[0], 2)
        self.assertEqual(R.shape[1], 2)

        a = np.array([1, 0])
        b = R @ a

        self.assertAlmostEqual(b[0], 0)
        self.assertAlmostEqual(b[1], 1)

    def test_rot270(self):
        alpha = pi + pi/2
        R = gm.rot2d(alpha)
        self.assertEqual(len(R.shape), 2)
        self.assertEqual(R.shape[0], 2)
        self.assertEqual(R.shape[1], 2)

        a = np.array([1, 0])
        b = R @ a

        self.assertAlmostEqual(b[0], 0)
        self.assertAlmostEqual(b[1], -1)

    def test_normal(self):
        left = np.array([-1, 0], np.float32)
        right = np.array([1, 0], np.float32)

        n = gm.get_2d_normal(left, right)
        self.assertEqual(len(n), 2)
        self.assertAlmostEqual(la.norm(n), 1)

        self.assertAlmostEqual(n[0], 0)
        self.assertAlmostEqual(n[1], 1)

    def eval_2d_rotation_upward(self, p):
        p = p.astype(np.float32)
        alpha = gm.get_2d_rotation_for_upward(p)
        R = gm.rot2d(alpha)
        p_norm = p / la.norm(p)
        up = R @ p_norm
        self.assertAlmostEqual(up[0], 0, places=4)
        self.assertAlmostEqual(up[1], 1, places=4)

    def test_2d_rotation_upward(self):
        p = np.array([1, 0])
        self.eval_2d_rotation_upward(p)

        p = np.array([1.7, 1.2])
        self.eval_2d_rotation_upward(p)

        p = np.array([-1.7, 1.2])
        self.eval_2d_rotation_upward(p)

        p = np.array([1.7, -1.2])
        self.eval_2d_rotation_upward(p)

        p = np.array([-1.7, -1.2])
        self.eval_2d_rotation_upward(p)


if __name__ == '__main__':
    unittest.main()
