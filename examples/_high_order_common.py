"""Shared runner helpers for high-order Lagrange convergence scripts.

This module intentionally contains no PDE-specific weak form.  Each equation
gets its own small entry script under ``examples/``; new equations can reuse
:func:`run_problem` by supplying their own ``problem_factory``.
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np


REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from MosicaFE import Mesh, compute_errors, estimate_convergence_rate, plot_convergence  # noqa: E402


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def exact(x):
    """Manufactured exact solution on the unit square."""
    x = np.asarray(x, dtype=float)
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def exact_grad(x):
    """Exact gradient of the manufactured solution."""
    x = np.asarray(x, dtype=float)
    return np.array(
        [
            np.pi * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
            np.pi * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
        ]
    )


def make_mesh(nx):
    """Create the triangular unit-square mesh for one refinement level."""
    return Mesh.rectangle(nx=nx, ny=nx, element_type="triangle")


def add_common_arguments(parser, *, include_wave_number=False):
    """Add degree/mesh/output arguments shared by all equation scripts."""
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Polynomial degrees P_k to run (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="Grid sizes nx=ny for the unit square (default: 4 8 16 32).",
    )
    if include_wave_number:
        parser.add_argument(
            "--wave-number",
            type=float,
            default=1.0,
            help="Helmholtz wave number k; keep k^2 < 2*pi^2 (~19.74).",
        )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not save convergence PNG files.",
    )


def validated_degrees_and_sizes(args):
    """Return sorted, validated ``(degrees, sizes)`` from parsed arguments."""
    degrees = sorted(set(int(d) for d in args.degrees))
    if any(d < 1 for d in degrees):
        raise SystemExit("degrees must be >= 1")

    sizes = sorted(set(int(n) for n in args.sizes))
    if any(n < 1 for n in sizes):
        raise SystemExit("sizes must be >= 1")

    return degrees, sizes


def run_problem(kind, degrees, sizes, problem_factory, *, save_plots=True):
    """Run one equation for all requested degrees and mesh sizes.

    Parameters
    ----------
    kind : str
        Short name used in filenames and printed output, e.g. ``"poisson"``.
    degrees : iterable of int
        Polynomial degrees to run.
    sizes : iterable of int
        Grid sizes ``nx=ny`` to run.
    problem_factory : callable
        ``problem_factory(mesh, degree) -> BasePhysics``.
    save_plots : bool
        Whether to write ``convergence_{kind}_P{degree}.png`` files.
    """
    rows = []

    print(f"\n{'=' * 78}")
    print(f"{kind.upper()}")
    print("=" * 78)

    for degree in degrees:
        h_values = []
        errors = {"L2": [], "H1": [], "Linf": []}

        print(f"\n--- {kind} P{degree} ---")
        for nx in sizes:
            mesh = make_mesh(nx)
            problem = problem_factory(mesh, degree)
            space = problem.space

            t0 = time.perf_counter()
            u = problem.solve(method="direct")
            solve_time = time.perf_counter() - t0

            norms = compute_errors(problem, u, exact, exact_grad=exact_grad)
            h = mesh.get_mesh_size()

            h_values.append(h)
            errors["L2"].append(norms["L2"])
            errors["H1"].append(norms["H1"])
            errors["Linf"].append(norms["Linf"])

            rows.append(
                {
                    "problem": kind,
                    "degree": degree,
                    "nx": nx,
                    "h": h,
                    "n_dofs": int(space.n_dofs),
                    "L2": norms["L2"],
                    "H1": norms["H1"],
                    "Linf": norms["Linf"],
                    "solve_time_s": solve_time,
                }
            )

            print(
                f"  n={nx:2d}  h={h:.4f}  dofs={space.n_dofs:5d}  "
                f"L2={norms['L2']:.4e}  H1={norms['H1']:.4e}  "
                f"Linf={norms['Linf']:.4e}  t={solve_time:.3f}s"
            )

        if len(sizes) >= 2:
            use_last = min(3, len(sizes))
            print(
                "  estimated rates (last 3 levels if available): "
                f"L2={estimate_convergence_rate(h_values, errors['L2'], use_last=use_last):.3f}, "
                f"H1={estimate_convergence_rate(h_values, errors['H1'], use_last=use_last):.3f}, "
                f"Linf={estimate_convergence_rate(h_values, errors['Linf'], use_last=use_last):.3f}"
            )

        if save_plots and len(sizes) >= 2:
            filename = os.path.join(OUT_DIR, f"convergence_{kind}_P{degree}.png")
            plot_convergence(
                h_values,
                errors,
                filename=filename,
                reference_rates=(1, 2, 3, degree, degree + 1),
            )

    csv_path = os.path.join(OUT_DIR, f"high_order_{kind}_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "problem",
            "degree",
            "nx",
            "h",
            "n_dofs",
            "L2",
            "H1",
            "Linf",
            "solve_time_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {csv_path}")
    return rows
