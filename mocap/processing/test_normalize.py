import sys
sys.path.insert(0, './../..')
import unittest
import numpy as np
import numpy.linalg as la
import mocap.processing.normalize as norm


class TestInsertRootNode(unittest.TestCase):

    def test_simple(self):

        seq = np.array([
            [
                (1, 0, 0),
                (-1, 0, 0)
            ],
            [
                (2, 0, 0),
                (0, 0, 0)
            ]
        ], np.float32)

        print(seq.shape)
        seq_n = norm.insert_root_node_as_avg(seq, 0, 1)
        assert len(seq_n.shape) == 3
        n, J, dim = seq_n.shape  # pylint: disable=unpacking-non-sequence
        self.assertEqual(n, 2)
        self.assertEqual(J, 3)
        self.assertEqual(dim, 3)

        a = la.norm(seq_n[0, 2] - np.array([0, 0, 0]))
        b = la.norm(seq_n[1, 2] - np.array([1, 0, 0]))
        self.assertAlmostEqual(a, 0)
        self.assertAlmostEqual(b, 0)


class TestTransforms(unittest.TestCase):

    def get_person(self):
        person = np.array([
            (10, 0, 10),  # root
            (15, 0, 10),  # right
            ( 5, 0, 10)  # left
        ], np.float32)
        return person

    def test_get_euclidean_transform(self):
        person = self.get_person()

        R, t = norm.get_euclidean_transform(person, j_root=0, j_right=1, j_left=2)

        for i in range(3):
            self.assertAlmostEqual(R[i, i], 1, places=4)

        self.assertAlmostEqual(t[0], -10, places=4)
        self.assertAlmostEqual(t[1], 0, places=4)
        self.assertAlmostEqual(t[2], -10, places=4)

    def test_rigid_transform(self):
        person = self.get_person()
        R, t = norm.get_euclidean_transform(person, j_root=0, j_right=1, j_left=2)
        new_person = norm.apply_euclidean_transform(person, R, t)
        for v in new_person[0]:
            self.assertAlmostEqual(v, 0, places=4)
        self.assertEqual(len(new_person[0]), 3)

        self.assertAlmostEqual(new_person[1,0], 5, places=4)
        self.assertAlmostEqual(new_person[2,0], -5, places=4)


if __name__ == '__main__':
    unittest.main()
