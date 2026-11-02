"""Run with: python -m unittest discover -s examples/RT0-Darcy -p 'test_*.py'."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('darcy_example', Path(__file__).with_name('rt0_darcy_2d.py'))
darcy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(darcy)


class DarcyExampleTests(unittest.TestCase):
    def test_refinement_preserves_nodes_and_linear_solution(self):
        mesh = darcy.make_rect_mesh(2, 2)
        for level in (0, 1, 2):
            result = darcy.solve_rt0(mesh, refinements=level, pressure=lambda x, y: 1-x)
            self.assertEqual(len(result['cells']), 8 * 4**level)
            np.testing.assert_array_equal(result['nodes'][:9], mesh[0])
            expected = 1-result['nodes'][result['cells']].mean(axis=1)[:, 0]
            np.testing.assert_allclose(result['P'], expected, atol=1e-12)
            self.assertLess(result['conservation'], 1e-12)

    def test_reject_invalid_refinements(self):
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                darcy.solve_rt0(darcy.rectangle_mesh(2), refinements=value)

    def test_tetrahedron_linear_pressure(self):
        mesh = (np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                np.array([[0, 1, 2, 3]]))
        result = darcy.solve_rt0(mesh, dim=3, pressure=lambda x, y, z: 1-x)
        np.testing.assert_allclose(result['P'], [.75], atol=1e-12)
        self.assertLess(result['conservation'], 1e-12)


if __name__ == '__main__':
    unittest.main()
