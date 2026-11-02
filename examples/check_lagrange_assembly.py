"""Compare assembly in memory and print a table to stdout.

Run: python -B examples/check_lagrange_assembly.py
No JSON, NPZ or image files are written by this script.
"""
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from MosicaFE import Mesh, FESpace
from MosicaFE._compat import to_dense
from MosicaFE.core.utils import simplex_jacobian, transform_gradient
from MosicaFE.spaces.lagrange import LagrangeSpace


class ReferenceLagrangeSpace(LagrangeSpace):
    """Pre-optimization matrix methods from commit 4cc162e, for comparison only.

    Basis, quadrature and geometry use MosicaFE's existing implementation.
    This class does not rely on Git history or external matrix snapshots.
    """

    def local_mass(self, cell_idx):
        pts, weights, basis, _ = self._quadrature_data(self.mass_quad_order())
        _, _, dets = self._element_geometry(cell_idx, pts)
        matrix = np.zeros((basis.shape[1], basis.shape[1]))
        for q in range(len(weights)):
            matrix += weights[q] * dets[q] * np.outer(basis[q], basis[q])
        return matrix

    def local_stiffness(self, cell_idx):
        pts, weights, _, grads_ref = self._quadrature_data(self.default_quad_order())
        if self._is_tensor():
            _, jac, dets = self._element_geometry(cell_idx, pts)
            grads = np.asarray([
                transform_gradient(grads_ref[q], jac[q]) for q in range(len(pts))
            ])
        else:
            jac, det = simplex_jacobian(self.mesh.get_cell_vertices(cell_idx))
            grads = grads_ref @ np.linalg.inv(jac)
            dets = np.full(len(weights), abs(det))
        return np.einsum("q,qia,qja->ij", weights * dets, grads, grads)

    def mass_matrix(self):
        return self._assemble(self.local_mass, symmetric=True)

    def stiffness_matrix(self):
        return self._assemble(self.local_stiffness, symmetric=True)


def cases():
    for n in (16, 32, 64):
        mesh = Mesh.rectangle(nx=n, ny=n)
        for p in (1, 2, 3):
            yield f"tri_n{n}_p{p}", mesh, p, True
    # 补充非规则几何、任意阶与其他维数的正确性检查，不计性能。
    tri = Mesh.rectangle(nx=3, ny=3)
    tri.nodes[5] += [0.035, -0.02]
    quad = Mesh.rectangle(nx=3, ny=3, element_type="quad")
    quad.nodes[5] += [0.035, -0.02]
    for name, mesh in (("irregular_tri", tri), ("distorted_quad", quad),
                       ("interval", Mesh.interval(nx=3)),
                       ("tet", Mesh.box(nx=1, ny=1, nz=1)),
                       ("hex", Mesh.box(nx=1, ny=1, nz=1, element_type="hex"))):
        for p in (1, 2, 3):
            yield f"{name}_p{p}", mesh, p, False


def measure(assemble, timed):
    matrix = assemble().tocsr()  # Warm reference caches before timing.
    samples = []
    if timed:
        for _ in range(5):
            start = time.perf_counter()
            matrix = assemble().tocsr()
            samples.append(time.perf_counter() - start)
    return matrix, samples


def check_analytic_and_fallback():
    mesh = Mesh.from_arrays([[0., 0.], [1., 0.], [0., 1.]], [[0, 1, 2]])
    space = FESpace.lagrange(mesh, 1)
    assert np.allclose(space.local_mass(0), (np.ones((3, 3)) + np.eye(3)) / 24)
    assert np.allclose(space.local_stiffness(0), [[1, -.5, -.5], [-.5, .5, 0], [-.5, 0, .5]])
    import MosicaFE._compat as compat
    available = compat.HAS_SCIPY
    try:
        compat.HAS_SCIPY = False
        assert np.allclose(space.mass_matrix(), space.local_mass(0))
        assert np.allclose(space.stiffness_matrix(), space.local_stiffness(0))
    finally:
        compat.HAS_SCIPY = available


def run():
    """Run paired assembly checks and return the records used for the table."""
    records = []
    for name, mesh, degree, timed in cases():
        reference = ReferenceLagrangeSpace(mesh, degree)
        space = FESpace.lagrange(mesh, degree)
        for kind in ("mass", "stiffness"):
            old, old_samples = measure(getattr(reference, kind + "_matrix"), timed)
            matrix, samples = measure(getattr(space, kind + "_matrix"), timed)
            error = matrix - old
            absolute = float(np.max(np.abs(error.data), initial=0))
            relative = float(scipy.sparse.linalg.norm(error) / scipy.sparse.linalg.norm(old))
            local = getattr(space, "local_" + kind)(0)
            local_error = float(np.max(np.abs(local - getattr(reference, "local_" + kind)(0))))
            symmetry = matrix - matrix.T
            assert absolute < 1e-10 and relative < 1e-11, (name, kind, absolute, relative)
            assert local_error < 1e-10, (name, kind, local_error)
            assert np.max(np.abs(symmetry.data), initial=0) < 1e-12
            if kind == "stiffness":
                assert np.max(np.abs(matrix @ np.ones(space.n_dofs))) < 1e-10
            else:
                assert abs(matrix.sum() - mesh.cell_measures().sum()) < 1e-10
            if not timed:
                eigenvalues = np.linalg.eigvalsh(to_dense(matrix))
                assert eigenvalues[0] > (0 if kind == "mass" else -1e-10)
            old_median = statistics.median(old_samples) if timed else None
            median = statistics.median(samples) if timed else None
            record = dict(case=name, matrix=kind, ndofs=space.n_dofs,
                          baseline_samples_s=old_samples, samples_s=samples,
                          baseline_median_s=old_median, median_s=median,
                          speedup=old_median / median if timed else None,
                          max_abs_error=absolute, relative_frobenius_error=relative,
                          local_max_abs_error=local_error)
            records.append(record)
    check_analytic_and_fallback()
    print("\nLagrange matrix assembly validation")
    print("=" * 116)
    print(f"{'case':22} {'matrix':10} {'ndofs':>8} {'old(ms)':>12} {'new(ms)':>12} "
          f"{'speedup':>8} {'abs_err':>10} {'rel_err':>10} {'local_err':>10}")
    print("-" * 116)
    for row in records:
        old = f"{row['baseline_median_s'] * 1000:.4f}" if row['baseline_median_s'] is not None else "-"
        new = f"{row['median_s'] * 1000:.4f}" if row['median_s'] is not None else "-"
        speed = f"{row['speedup']:.2f}x" if row['speedup'] is not None else "-"
        print(f"{row['case']:22} {row['matrix']:10} {row['ndofs']:8d} "
              f"{old:>12} {new:>12} {speed:>8} "
              f"{row['max_abs_error']:10.3e} {row['relative_frobenius_error']:10.3e} "
              f"{row['local_max_abs_error']:10.3e}")
    print("=" * 116)
    print(f"Validation passed: {len(records)} matrix comparisons, "
          "P1 analytic and NumPy fallback checks")
    print("Maximum errors: "
          f"absolute={max(r['max_abs_error'] for r in records):.3e}, "
          f"relative={max(r['relative_frobenius_error'] for r in records):.3e}, "
          f"local={max(r['local_max_abs_error'] for r in records):.3e}")
    return records


if __name__ == "__main__":
    run()
