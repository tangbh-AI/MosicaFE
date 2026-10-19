"""Enhanced conforming VEM machinery; used internally by VEMSpace.

Face moments use a canonical, shared orthonormal frame. In each enhanced
space, moments of degrees k-1 and k equal those of the energy projection;
the lower moments remain the actual degrees of freedom. This makes the
full L2 projector computable without evaluating virtual basis functions.
"""
from __future__ import annotations

import numpy as np

from ..core.mesh import Mesh
from ..core.quadrature import cell_quadrature
from ..core.utils import cell_centroid, cell_diameter


class HighOrderVEMMixin:
    @staticmethod
    def _complete_l2_projection(p):
        # Scaled monomials are still ill-conditioned as k increases. Enforce
        # the exact polynomial identity after floating-point linear solves.
        D = p['D']
        left = np.linalg.pinv(D)
        identity = np.eye(D.shape[1])
        p['Pi'] += (identity-p['Pi'] @ D) @ left
        C = p['H'] @ p['Pi']
        nm = len(p['mom_multi'])
        start = p['moment_start']
        C[:nm] = 0.0
        C[np.arange(nm), start + np.arange(nm)] = p['measure']
        scale = np.sqrt(np.diag(p['H']))
        equilibrated = p['H']/scale[:, None]/scale[None, :]
        Pi0 = np.linalg.solve(equilibrated, C/scale[:, None])/scale[:, None]
        Pi0 += (identity-Pi0 @ D) @ left
        p['Pi0'] = Pi0

    def _high_quadrature(self, cell_idx, order):
        if self.mesh.dim == 2:
            return self._fan_quadrature(self.mesh.get_cell_vertices(cell_idx), order)
        return cell_quadrature(self.mesh, cell_idx, order)

    def boundary_dofs(self):
        if self.degree == 1:
            return super().boundary_dofs()
        # Testing whether *all vertices* are boundary vertices incorrectly
        # constrains interior diagonals. Test support against actual facets.
        boundary = [set(f) for f in self.mesh.boundary_facets]
        by_vertex = {}
        for face in boundary:
            for v in face:
                by_vertex.setdefault(v, []).append(face)
        out = []
        for i, support in enumerate(self.dof_support):
            if not support or self.dof_is_interior[i]:
                continue
            s = set(support)
            if any(s <= face for face in by_vertex.get(support[0], [])):
                out.append(i)
        return np.asarray(out, dtype=np.int64)

    def boundary_values(self, f, dofs):
        if self.degree == 1 or self.mesh.dim == 2:
            return self._evaluate_function(f, self.dof_coords[dofs])
        # A face moment is an integral, not the value at its representative point.
        vals = np.zeros(self.n_dofs)
        point = self._point_dofs
        vals[point] = self._evaluate_function(f, self.dof_coords[point])
        wanted = set(int(i) for i in dofs)
        for face in self._high_faces.values():
            if any(int(i) in wanted for i in face['moment_dofs']):
                self._set_face_moments(vals, face, f)
        return vals[dofs]

    def _build_dof_map_3d(self):
        from .vem import _monomial_indices, VEMSpace
        k, mesh = self.degree, self.mesh
        book, supports, interiors, locations = {}, [], [], []

        def reg(key, support, inside, point):
            if key not in book:
                book[key] = len(supports)
                supports.append(tuple(support))
                interiors.append(inside)
                locations.append(np.asarray(point))
            return book[key]

        self._high_faces, self._cell_high_faces = {}, []
        self.dof_map = []
        point_dofs = set()
        for c, cell in enumerate(mesh.elements):
            local, face_records = [], []
            for v in cell:
                local.append(reg(('v', int(v)), (int(v),), False, mesh.nodes[v]))
                point_dofs.add(local[-1])
            faces = self._cell_faces(c)
            edges = sorted({tuple(sorted((int(cell[f[i]]), int(cell[f[(i+1) % len(f)]]))))
                            for f, _, _ in faces for i in range(len(f))})
            for a, b in edges:
                for j in range(1, k):
                    idx = reg(('e', a, b, j), (a, b), False,
                              (1-j/k)*mesh.nodes[a] + (j/k)*mesh.nodes[b])
                    local.append(idx)
                    point_dofs.add(idx)
            for f, normal, area in faces:
                vertices = [int(cell[i]) for i in f]
                key = tuple(sorted(vertices))
                if key not in self._high_faces:
                    # Canonical cyclic ordering: minimum vertex first; minimum
                    # of its two neighbours second. Never sort a polygon cycle.
                    j = vertices.index(min(vertices))
                    cycle = vertices[j:] + vertices[:j]
                    if cycle[-1] < cycle[1]:
                        cycle = cycle[:1] + cycle[:0:-1]
                    xyz = mesh.nodes[cycle]
                    origin = xyz.mean(axis=0)
                    e1 = xyz[1]-xyz[0]
                    e1 /= np.linalg.norm(e1)
                    n = np.cross(e1, xyz[-1]-xyz[0])
                    n /= np.linalg.norm(n)
                    frame = np.column_stack((e1, np.cross(n, e1)))
                    xy = (xyz-origin) @ frame
                    fm = Mesh(xy, [list(range(len(cycle)))], element_type='polygon')
                    fs = VEMSpace(fm, degree=k, stabilization=self.stabilization)
                    ordered = [cycle[int(i)] for i in fm.elements[0]]
                    fd = [reg(('v', v), (v,), False, mesh.nodes[v]) for v in ordered]
                    for a, b in zip(ordered, ordered[1:]+ordered[:1]):
                        lo, hi = sorted((a, b))
                        for j in range(1, k):
                            gj = j if a < b else k-j
                            fd.append(reg(('e', lo, hi, gj), (lo, hi), False,
                                          (1-j/k)*mesh.nodes[a]+(j/k)*mesh.nodes[b]))
                    md = []
                    for j in range(len(_monomial_indices(2, k-2))):
                        md.append(reg(('f', key, j), key, False, origin))
                    fd.extend(md)
                    self._high_faces[key] = dict(space=fs, origin=origin, frame=frame,
                                                 dofs=np.array(fd), moment_dofs=np.array(md))
                face = self._high_faces[key]
                local.extend(face['moment_dofs'])
                face_records.append((face, normal, area))
            for j in range(len(_monomial_indices(3, k-2))):
                local.append(reg(('c', c, j), (), True,
                                 cell_centroid(mesh.get_cell_vertices(c))))
            self.dof_map.append(np.asarray(local, dtype=np.int64))
            self._cell_high_faces.append(face_records)
        self.n_dofs = len(supports)
        self.dof_support = supports
        self.dof_is_interior = np.asarray(interiors, dtype=bool)
        self.dof_coords = np.asarray(locations)
        self._point_dofs = np.array(sorted(point_dofs), dtype=np.int64)
        self._point_set = point_dofs

    def _projection_3d(self, cell_idx):
        from .vem import (_monomial_indices, _scaled_monomial_values,
                          _scaled_monomial_grad)
        if cell_idx in self._proj_cache:
            return self._proj_cache[cell_idx]
        k = self.degree
        coords = self.mesh.get_cell_vertices(cell_idx)
        centroid, h = cell_centroid(coords), cell_diameter(coords)
        multi, mm = _monomial_indices(3, k), _monomial_indices(3, k-2)
        pts, w = self._high_quadrature(cell_idx, 2*k)
        measure = w.sum()
        mono = _scaled_monomial_values(multi, pts, centroid, h)
        grad = _scaled_monomial_grad(multi, pts, centroid, h)
        H = mono.T @ (w[:, None]*mono)
        B = np.einsum('qad,qbd,q->ab', grad, grad, w)
        dofs = self.get_cell_dofs(cell_idx)
        lookup = {int(d): i for i, d in enumerate(dofs)}
        nm, nk, nd = len(mm), len(multi), len(dofs)
        start = nd-nm
        D, R = np.zeros((nd, nk)), np.zeros((nk, nd))
        point_rows = [i for i, d in enumerate(dofs) if int(d) in self._point_set]
        D[point_rows] = _scaled_monomial_values(
            multi, self.dof_coords[dofs[point_rows]], centroid, h)
        for face, normal, area in self._cell_high_faces[cell_idx]:
            fs = face['space']
            fp = fs._projection_high(0)
            xy, fw = fs._high_quadrature(0, 2*k)
            xyz = face['origin'] + xy @ face['frame'].T
            fm = _scaled_monomial_values(fp['multi'], xy, fp['centroid'], fp['h'])
            fq = fm[:, :len(fp['mom_multi'])]
            rows = [lookup[int(d)] for d in face['dofs']]
            face_start = fp['moment_start']
            D[rows[face_start:]] = fq.T @ (fw[:, None] *
                _scaled_monomial_values(multi, xyz, centroid, h)) / area
            dn = _scaled_monomial_grad(multi, xyz, centroid, h) @ normal
            R[:, rows] += dn.T @ (fw[:, None] * (fm @ fp['Pi0']))
        D[start:] = H[:nm] / measure
        mom_idx = {a: i for i, a in enumerate(mm)}
        for i, a in enumerate(multi):
            for d in range(3):
                if a[d] >= 2:
                    b = list(a)
                    b[d] -= 2
                    R[i, start+mom_idx[tuple(b)]] -= measure*a[d]*(a[d]-1)/h**2
        Pi = np.zeros((nk, nd))
        Pi[1:] = np.linalg.solve(B[1:, 1:], R[1:])
        Pi[0, start] = 1.0
        Pi[0] -= (H[0, 1:]/measure) @ Pi[1:]
        p = dict(multi=multi, mom_multi=mm, centroid=centroid, h=h, measure=measure,
                 coords=coords, B=B, H=H, D=D, Pi=Pi, moment_start=start)
        self._complete_l2_projection(p)
        self._proj_cache[cell_idx] = p
        return p

    def _set_face_moments(self, vals, face, f):
        from .vem import _scaled_monomial_values
        fs = face['space']
        p = fs._projection_high(0)
        xy, w = fs._high_quadrature(0, 2*self.degree+2)
        xyz = face['origin'] + xy @ face['frame'].T
        q = _scaled_monomial_values(p['mom_multi'], xy, p['centroid'], p['h'])
        vals[face['moment_dofs']] = q.T @ (w*self._evaluate_function(f, xyz))/p['measure']

    def _interpolate_3d(self, f):
        from .vem import _scaled_monomial_values, _monomial_indices
        vals = np.zeros(self.n_dofs)
        vals[self._point_dofs] = self._evaluate_function(f, self.dof_coords[self._point_dofs])
        for face in self._high_faces.values():
            self._set_face_moments(vals, face, f)
        for c in range(self.mesh.n_cells):
            coords = self.mesh.get_cell_vertices(c)
            centroid, h = cell_centroid(coords), cell_diameter(coords)
            pts, w = self._high_quadrature(c, 2*self.degree+2)
            mm = _monomial_indices(3, self.degree-2)
            q = _scaled_monomial_values(mm, pts, centroid, h)
            vals[self.dof_map[c][-len(mm):]] = q.T @ (w*self._evaluate_function(f, pts))/w.sum()
        return vals
