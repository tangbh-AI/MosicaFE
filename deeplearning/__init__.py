"""PyTorch-only optional PDE learning; importing MosicaFE itself does not load torch."""
try:
    import torch as _torch
except ImportError as exc:
    raise ImportError("MosicaFE.deeplearning requires PyTorch; install mosicafe[deeplearning] in AI4Sci") from exc

from .api import solve_pinn, solve_supervised
from .autodiff import derivative, divergence, gradient, laplacian
from .conditions import DirichletBC, InitialCondition, NeumannBC, PointCondition
from .data import PDEData
from .domain import Domain
from .equations import EllipticResidual, HeatResidual
from .networks import MLP
from .problems import PINNProblem, SupervisedProblem
from .solution import NeuralSolution
from .trainer import Trainer, TrainingConfig

__all__ = ["Domain", "MLP", "PDEData", "PINNProblem", "SupervisedProblem", "Trainer",
           "TrainingConfig", "NeuralSolution", "DirichletBC", "InitialCondition", "NeumannBC",
           "PointCondition", "EllipticResidual", "HeatResidual", "gradient", "derivative",
           "laplacian", "divergence", "solve_pinn", "solve_supervised"]
