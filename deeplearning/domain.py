"""Box sampling and mesh quadrature; reuse the actual MosicaFE geometry."""
from __future__ import annotations

import numpy as np

from ..core.quadrature import cell_quadrature, simplex_quadrature


class Domain:
    """A box or an existing Mesh. Mesh domains use positive quadrature weights."""

    def __init__(self, bounds, mesh=None):
        self.bounds = np.asarray(bounds, dtype=float)
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 2
                or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 1] <= self.bounds[:, 0])):
            raise ValueError("bounds must be finite (dim, 2) increasing intervals")
        self.dim = len(self.bounds)
        self.mesh = mesh

    @classmethod
    def interval(cls, a=0., b=1.):
        return cls([[a, b]])

    @classmethod
    def rectangle(cls, xmin=0., xmax=1., ymin=0., ymax=1.):
        return cls([[xmin, xmax], [ymin, ymax]])

    @classmethod
    def box(cls, bounds):
        return cls(bounds)

    @classmethod
    def from_mesh(cls, mesh):
        return cls(np.column_stack((mesh.nodes.min(axis=0), mesh.nodes.max(axis=0))), mesh=mesh)

    def interior(self, n=512, seed=0):
        """(points, weights). For meshes n is ignored; all order-2 cell points are used."""
        if self.mesh is not None:
            rules = [cell_quadrature(self.mesh, c, order=2) for c in range(self.mesh.n_cells)]
            x, w = np.vstack([r[0] for r in rules]), np.concatenate([r[1] for r in rules])
            if np.any(w <= 0):
                raise ValueError("Mesh sampling requires positive quadrature weights")
            return x, w
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        rng = np.random.default_rng(seed)
        x = rng.random((n, self.dim))
        return self.bounds[:, 0] + x * np.diff(self.bounds, axis=1).ravel(), np.ones(n)

    def boundary(self, n=128, seed=0):
        """(points, surface weights); mesh facets, never the mesh bounding box."""
        if self.mesh is not None:
            points, weights = [], []
            for facet in self.mesh.boundary_facets:
                v = self.mesh.nodes[list(facet)]
                if self.dim == 1:
                    p, w = v, np.ones(len(v))
                elif len(v) <= 3:
                    p, w = simplex_quadrature(v, order=2)
                else:
                    # Facet topology stores sorted indices; restore planar cyclic order.
                    center = v.mean(axis=0)
                    _, _, basis = np.linalg.svd(v - center, full_matrices=False)
                    uv = (v - center) @ basis[:2].T
                    v = v[np.argsort(np.arctan2(uv[:, 1], uv[:, 0]))]
                    rules = [simplex_quadrature([center, v[i], v[(i + 1) % len(v)]], order=2)
                             for i in range(len(v))]
                    p, w = np.vstack([r[0] for r in rules]), np.concatenate([r[1] for r in rules])
                points.append(p)
                weights.append(w)
            if not points:
                raise ValueError("Mesh has no boundary facets")
            return np.vstack(points), np.concatenate(weights)
        if not isinstance(n, int) or n < 2 * self.dim:
            raise ValueError("n must be at least 2 * dim so every box face is sampled")
        x, _ = self.interior(n=n, seed=seed)
        face = np.arange(n) % (2 * self.dim)
        widths = np.diff(self.bounds, axis=1).ravel()
        w = np.empty(n)
        for f in range(2 * self.dim):
            mask = face == f
            axis, side = divmod(f, 2)
            x[mask, axis] = self.bounds[axis, side]
            area = np.prod(np.delete(widths, axis))
            w[mask] = area / mask.sum()
        return x, w


def as_domain(domain):
    from ..core.mesh import Mesh
    if isinstance(domain, Domain):
        return domain
    if isinstance(domain, Mesh):
        return Domain.from_mesh(domain)
    return Domain(domain)
