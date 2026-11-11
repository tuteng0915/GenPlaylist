import itertools
import unittest
import numpy as np
from oas import assignment_oas, normalized


class AssignmentTests(unittest.TestCase):
    def test_identity_permutation_and_scaling(self):
        x = np.eye(5)[None]
        r = assignment_oas(x * 7, x[:, [4, 0, 1, 2, 3]] * .2)
        np.testing.assert_allclose(r['similarity'], 1)
        np.testing.assert_allclose(r['distance'], 0)
        np.testing.assert_array_equal(r['target_positions'], [[1, 2, 3, 4, 0]])

    def test_duplicates_cannot_reuse_a_target(self):
        p = np.repeat(np.eye(5)[None, :1], 5, axis=1)
        r = assignment_oas(p, np.eye(5)[None])
        np.testing.assert_allclose(r['similarity'], .2)

    def test_brute_force_and_official_distance(self):
        rng = np.random.default_rng(20261111)
        p, t = rng.normal(size=(2, 11, 5, 9))
        r = assignment_oas(p, t)
        for n, (a, b) in enumerate(zip(normalized(p), normalized(t))):
            s = a @ b.T
            brute = max(np.mean(s[np.arange(5), perm]) for perm in itertools.permutations(range(5)))
            self.assertAlmostEqual(r['similarity'][n], brute, places=12)
            self.assertAlmostEqual(r['distance'][n], 1-brute, places=12)
        np.testing.assert_allclose(assignment_oas(p[:, ::-1], t[:, ::-1])['similarity'], r['similarity'])

    def test_invalid_inputs(self):
        for x in [np.zeros((1, 5, 5)), np.full((1, 5, 5), np.nan), np.ones((1, 4, 5))]:
            with self.assertRaises(ValueError):
                assignment_oas(x, x)


if __name__ == '__main__':
    unittest.main()
