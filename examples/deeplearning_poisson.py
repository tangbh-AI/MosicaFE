"""2D Poisson PINN main program, optional resume, integral errors and FE plotting."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from MosicaFE import Mesh, FESpace, plot_solution
from MosicaFE.deeplearning import solve_pinn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2000, help="Total target epochs, including resumed epochs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    exact = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    source = lambda x: 2 * np.pi ** 2 * exact(x)

    mesh = Mesh.rectangle(nx=8, ny=8)
    u = solve_pinn(mesh, f=source, epochs=args.epochs, lr=.003, device=args.device,
                   run_dir=args.run_dir, resume_from=args.resume)
    space = FESpace.lagrange(mesh, degree=2)
    errors = u.compute_errors(exact, lambda x: np.pi * np.array([
        np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]),
        np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])]))
    print("Poisson PINN integral errors:", errors)
    np.save(u.run_dir / "nodal_prediction.npy", u(mesh.nodes))
    plot_solution(u.to_fe(space), mesh, space, filename=str(u.run_dir / "solution.png"))
    print("Run directory:", u.run_dir)
    print("Checkpoint:", u.run_dir / "last.pt")


if __name__ == "__main__":
    main()
