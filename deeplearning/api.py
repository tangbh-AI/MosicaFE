"""Short main programs: domain -> network -> problem -> training -> solution."""
from __future__ import annotations

from .domain import as_domain
from .equations import EllipticResidual, HeatResidual
from .networks import MLP
from .problems import PINNProblem, SupervisedProblem


def solve_pinn(domain, *, equation="poisson", f=0., g=0., kappa=1., reaction=0., k=1.,
               network=None, hidden_layers=(32, 32, 32), backend="numpy", residual=None,
               config=None, resume_from=None, conditions=None, n_interior=512, n_boundary=128,
               seed=0, data=None, data_weight=1., residual_weight=1., boundary_weight=10.,
               time_axis=-1, callback=None, **training_options):
    domain = as_domain(domain)
    if network is None:
        network = MLP(domain.dim, hidden_layers=hidden_layers, bounds=domain.bounds, seed=seed)
    if residual is None:
        if equation == "poisson":
            residual = EllipticResidual(f, kappa, 0., backend)
        elif equation == "reaction_diffusion":
            residual = EllipticResidual(f, kappa, reaction, backend)
        elif equation == "helmholtz":
            residual = EllipticResidual(f, kappa, -k ** 2, backend)
        elif equation == "heat":
            if conditions is None:
                raise ValueError("Heat equation requires explicit spatial BCs and initial conditions")
            residual = HeatResidual(f, kappa, time_axis, backend)
        else:
            raise ValueError("equation must be poisson/reaction_diffusion/helmholtz/heat, or provide residual")
    obj = PINNProblem(domain, network, residual=residual, g=g, backend=backend, conditions=conditions,
                      n_interior=n_interior, n_boundary=n_boundary, seed=seed,
                      residual_weight=residual_weight, boundary_weight=boundary_weight,
                      data=data, data_weight=data_weight)
    return obj.solve(config, resume_from=resume_from, callback=callback, **training_options)


def solve_supervised(data, *, network=None, domain=None, hidden_layers=(32, 32, 32),
                     seed=0, config=None, resume_from=None, callback=None, **training_options):
    if network is None:
        bounds = None if domain is None else as_domain(domain).bounds
        network = MLP(data.points.shape[1], data.values.shape[1], hidden_layers, bounds=bounds, seed=seed)
    obj = SupervisedProblem(network, data, domain)
    return obj.solve(config, resume_from=resume_from, callback=callback, **training_options)
