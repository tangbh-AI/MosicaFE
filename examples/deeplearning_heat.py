"""Space-time heat PINN: u_t-u_xx=0, u(x,0)=sin(pi*x), u(0,t)=u(1,t)=0."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from MosicaFE.deeplearning import Domain, DirichletBC, InitialCondition, solve_pinn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    domain = Domain.box([[0., 1.], [0., .5]])  # coordinates: (x,t)
    t = np.linspace(0, .5, 64)
    bx = np.vstack([np.column_stack([np.zeros_like(t), t]), np.column_stack([np.ones_like(t), t])])
    ix = np.column_stack([np.linspace(0, 1, 64), np.zeros(64)])
    conditions = [DirichletBC(bx, 0., weight=10.),
                  InitialCondition(ix, lambda p: np.sin(np.pi * p[0]), weight=10.)]
    u = solve_pinn(domain, equation="heat", conditions=conditions, epochs=args.epochs,
                   lr=.003, run_dir=args.run_dir, resume_from=args.resume, device=args.device)
    x, t = np.meshgrid(np.linspace(0, 1, 51), np.linspace(0, .5, 31))
    points = np.column_stack([x.ravel(), t.ravel()])
    exact = np.sin(np.pi * points[:, 0]) * np.exp(-np.pi ** 2 * points[:, 1])
    print("Space-time held-out RMSE:", np.sqrt(np.mean((u(points) - exact) ** 2)))
    np.savez(u.run_dir / "heat_prediction.npz", points=points, prediction=u(points), exact=exact)
    print("Run directory:", u.run_dir)


if __name__ == "__main__":
    main()
