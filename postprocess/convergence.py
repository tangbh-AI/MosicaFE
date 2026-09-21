"""收敛性研究（guidebook 第 8.3 节）。

.. code-block:: python

   from MosicaFE.postprocess import convergence_study, estimate_convergence_rate

   def mesh_factory(level):
       return Mesh.rectangle(nx=4 * 2**level, ny=4 * 2**level)

   def problem_factory(mesh):
       V = FESpace.lagrange(mesh, degree=2)
       return PoissonProblem(V, f=source, g=lambda x: 0.0)

   h, errors = convergence_study(mesh_factory, problem_factory,
                                 exact_solution, exact_gradient,
                                 refinement_levels=5, plot=True)
"""

from __future__ import annotations

import numpy as np

from .error import compute_errors, estimate_convergence_rate, rate_table

__all__ = ["convergence_study", "estimate_convergence_rate", "rate_table"]


def convergence_study(
    mesh_factory,
    problem_factory,
    exact_solution,
    exact_gradient=None,
    refinement_levels: int = 5,
    plot: bool = False,
    filename: str = None,
    show: bool = False,
    solver: str = "auto",
    verbose: bool = True,
    callback=None,
):
    """逐层加密网格，记录 ``h`` 与 L2/H1/L∞ 误差。

    Parameters
    ----------
    mesh_factory : callable
        ``mesh_factory(level) -> Mesh``。
    problem_factory : callable
        ``problem_factory(mesh) -> BasePhysics``。
    exact_solution : callable
    exact_gradient : callable, optional
    refinement_levels : int
    plot : bool
        是否画 log-log 收敛曲线。
    callback : callable, optional
        ``callback(level, mesh, problem, solution, norms)``，用于自定义记录。

    Returns
    -------
    h_values : (n,) ndarray
    errors : dict
        ``{"L2": array, "H1": array, "Linf": array}``
    """
    h_values = []
    errors = {"L2": [], "H1": [], "Linf": []}

    for level in range(int(refinement_levels)):
        mesh = mesh_factory(level)
        problem = problem_factory(mesh)
        solution = problem.solve(method=solver)
        norms = compute_errors(problem, solution, exact_solution, exact_gradient)

        h_values.append(mesh.get_mesh_size())
        for key in errors:
            errors[key].append(norms.get(key, np.nan))

        if verbose:
            print(
                f"  level {level}: h={h_values[-1]:.5f}  "
                f"L2={norms['L2']:.4e}  H1={norms['H1']:.4e}  Linf={norms['Linf']:.4e}"
            )
        if callback is not None:
            callback(level, mesh, problem, solution, norms)

    h_values = np.asarray(h_values, dtype=float)
    errors = {k: np.asarray(v, dtype=float) for k, v in errors.items()}

    if verbose:
        print("\n收敛表：")
        print(rate_table(h_values, errors))
        print("\n估计收敛阶（用最后 3 层）：")
        for key in ("L2", "H1"):
            vals = errors[key]
            if np.all(np.isfinite(vals)) and np.any(vals > 0):
                rate = estimate_convergence_rate(h_values, vals, use_last=3)
                print(f"  {key:>4s}: {rate:.3f}")

    if plot:
        from .visualization import plot_convergence

        plot_convergence(h_values, errors, filename=filename, show=show)

    return h_values, errors
