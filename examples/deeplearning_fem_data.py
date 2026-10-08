"""Train an ordinary fully connected network on a traditional FEM PDE solution."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from MosicaFE import Mesh, FESpace, PoissonProblem, plot_solution
from MosicaFE.deeplearning import PDEData, solve_supervised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    exact = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    source = lambda x: 2 * np.pi ** 2 * exact(x)

    mesh = Mesh.rectangle(nx=8, ny=8)
    space = FESpace.lagrange(mesh, degree=2)
    uh = PoissonProblem(space, f=source).solve()
    data = PDEData.from_fe(space, uh)
    u = solve_supervised(data, domain=mesh, epochs=args.epochs, lr=.003,
                         device=args.device, run_dir=args.run_dir, resume_from=args.resume)
    dense = Mesh.rectangle(nx=24, ny=24).nodes
    print("Held-out nodal RMSE:", np.sqrt(np.mean((u(dense) - np.array([exact(p) for p in dense])) ** 2)))
    plot_solution(u.to_fe(space), mesh, space, filename=str(u.run_dir / "solution.png"))
    np.save(u.run_dir / "traditional_dofs.npy", uh)
    print("Run directory:", u.run_dir)


if __name__ == "__main__":
    main()
