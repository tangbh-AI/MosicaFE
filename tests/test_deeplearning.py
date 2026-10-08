"""Numerical, interoperability and persistence checks for the optional torch module."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from MosicaFE import FESpace, Mesh, PoissonProblem
from MosicaFE.deeplearning import (
    Domain, MLP, PDEData, PINNProblem, SupervisedProblem, NeuralSolution,
    TrainingConfig, Trainer, EllipticResidual, HeatResidual, PointCondition,
    DirichletBC, NeumannBC, InitialCondition, derivative, gradient, laplacian,
    solve_pinn, solve_supervised,
)


def test_optional_import():
    script = "import sys; import MosicaFE; assert 'torch' not in sys.modules; from MosicaFE import Mesh; assert Mesh.interval().dim == 1"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    subprocess.run([sys.executable, "-c", script], check=True, env=env, capture_output=True)


def test_autodiff_physical_coordinates_and_constant_derivative():
    x = torch.tensor([[.2, .4], [.3, .6]], dtype=torch.float64, requires_grad=True)
    u = (x[:, 0] ** 3 + x[:, 1] ** 2)[:, None]
    torch.testing.assert_close(gradient(u, x), torch.stack([3 * x[:, 0] ** 2, 2 * x[:, 1]], dim=1))
    torch.testing.assert_close(laplacian(u, x), (6 * x[:, 0] + 2)[:, None])
    torch.testing.assert_close(derivative(2 * x[:, :1] + 1, x, 0, 2), torch.zeros((2, 1), dtype=x.dtype))
    model = MLP(1, hidden_layers=(), bounds=[[2., 6.]]).double()
    with torch.no_grad():
        model.layers[0].weight.fill_(2.)
    y = torch.tensor([[3.], [4.]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(gradient(model(y), y), torch.ones_like(y))
    # Differentiation must continue to the network's parameters.
    laplacian(MLP(2).double()(x), x).square().sum().backward()


def test_residuals_variable_diffusion_heat_and_sign():
    x = torch.tensor([[.2, .3], [.7, .6]], dtype=torch.float64, requires_grad=True)
    u = x[:, :1] ** 2 + x[:, 1:] ** 2
    residual = EllipticResidual(f=lambda x: -4 - 6 * x[:, 0:1],
                                kappa=lambda x: 1 + x[:, 0:1], backend="torch")
    # div((1+x) grad(x^2+y^2)) = 4+6x, so use the independent analytic formula.
    torch.testing.assert_close(residual(x, u), torch.zeros_like(u), atol=1.e-12, rtol=0)
    helmholtz = EllipticResidual(f=lambda p: -4 - 9 * (p[0] ** 2 + p[1] ** 2), reaction=-9.)
    torch.testing.assert_close(helmholtz(x, u), torch.zeros_like(u), atol=1.e-12, rtol=0)
    heat = HeatResidual()
    h = torch.sin(torch.pi * x[:, :1]) * torch.exp(-torch.pi ** 2 * x[:, 1:])
    torch.testing.assert_close(heat(x, h), torch.zeros_like(h), atol=1.e-12, rtol=0)
    with pytest.raises(ValueError, match="Variable kappa"):
        EllipticResidual(kappa=lambda p: 1 + p[0])


def test_mesh_domain_uses_geometry_and_surface_measure():
    mesh = Mesh.from_arrays([[0., 0.], [2., 0.], [0., 1.]], [[0, 1, 2]], element_type="triangle")
    domain = Domain.from_mesh(mesh)
    points, w = domain.interior()
    assert np.all(points[:, 0] / 2 + points[:, 1] <= 1 + 1.e-12)
    assert w.sum() == pytest.approx(1.)
    _, bw = domain.boundary()
    assert bw.sum() == pytest.approx(3 + np.sqrt(5))
    box = Domain.rectangle(xmax=3., ymax=2.)
    bx, bw = box.boundary(33)
    assert bw.sum() == pytest.approx(10.)
    assert np.all(np.any((bx == box.bounds[:, 0]) | (bx == box.bounds[:, 1]), axis=1))
    mesh3 = Mesh.box(nx=1, ny=1, nz=1, element_type="hex")
    _, bw3 = Domain.from_mesh(mesh3).boundary()
    assert bw3.sum() == pytest.approx(6.)


def test_conditions_and_pinn_validation():
    model = MLP(1, hidden_layers=()).double()
    with torch.no_grad():
        model.layers[0].weight.fill_(2)
        model.layers[0].bias.fill_(1)
    c = NeumannBC([[0.], [1.]], [[-1.], [1.]], [-2., 2.])
    assert c.loss(model, device="cpu", dtype=torch.float64).item() == pytest.approx(0.)
    ic = InitialCondition([[0.], [1.]], [1., 3.])
    assert ic.loss(model, device="cpu", dtype=torch.float64).item() == pytest.approx(0.)
    robin = PointCondition([[0.]], 3., operator=lambda x, u: u + gradient(u, x))
    assert robin.loss(model, device="cpu", dtype=torch.float64).item() == pytest.approx(0.)
    with pytest.raises(ValueError, match="smooth"):
        PINNProblem(Domain.interval(), MLP(1, activation="relu"))
    with pytest.raises(ValueError, match="explicit"):
        solve_pinn(Domain.rectangle(), equation="heat")
    with pytest.raises(ValueError, match="Data"):
        PDEData([[0.]], [1., 2.])


def test_fe_and_mixed_exchange():
    mesh = Mesh.rectangle(nx=2, ny=2)
    model = MLP(2, hidden_layers=()).double()
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.tensor([[2., -3.]], dtype=torch.float64))
        model.layers[0].bias.fill_(1)
    sol = NeuralSolution(model, Domain.from_mesh(mesh))
    exact = lambda p: 1 + 2 * p[0] - 3 * p[1]
    for space in (FESpace.lagrange(mesh, 1), FESpace.lagrange(mesh, 2), FESpace.vem(mesh)):
        u = sol.to_fe(space)
        np.testing.assert_allclose(u, space.interpolate(exact), atol=1.e-12)
        data = PDEData.from_fe(space, u)
        np.testing.assert_allclose(data.values[:, 0], [exact(p) for p in data.points], atol=1.e-12)
        assert data.weights.sum() == pytest.approx(1.)
    norms = sol.compute_errors(exact, lambda p: np.array([2., -3.]))
    assert norms["L2"] < 1.e-12 and norms["H1"] < 1.e-12
    rt = FESpace.rt0(mesh)
    p = sol.to_pressure(rt)
    q = sol.to_flux(rt)
    np.testing.assert_allclose(p, rt.interpolate_pressure(exact))
    np.testing.assert_allclose(q, rt.interpolate_flux(lambda p: [-2., 3.]))
    local_q = np.vstack([rt.local_flux_values(c, q, global_values=True) for c in range(mesh.n_cells)])
    pressure_data = PDEData.from_fe(rt, (local_q, p), field="pressure")
    flux_data = PDEData.from_fe(rt, (local_q, p), field="flux")
    assert pressure_data.values.shape[1] == 1
    np.testing.assert_allclose(flux_data.values, np.tile([-2., 3.], (len(flux_data.points), 1)), atol=1.e-12)
    with pytest.raises(ValueError, match="RT0"):
        sol.to_fe(rt)


@pytest.mark.parametrize("dim,single_cell", [(2, False), (3, False), (2, True), (3, True)])
def test_rt0_constant_flux_reconstruction(dim, single_cell):
    if single_cell:
        nodes = np.vstack([np.zeros(dim), np.eye(dim)])
        mesh = Mesh.from_arrays(nodes, [np.arange(dim + 1)], element_type="triangle" if dim == 2 else "tet")
    else:
        mesh = Mesh.rectangle(nx=3, ny=2) if dim == 2 else Mesh.box(nx=2, ny=2, nz=2)
    space = FESpace.rt0(mesh)
    constant = np.arange(1, dim + 1, dtype=float)
    phi = space.interpolate_flux(lambda p: constant)
    # The integral of a constant field is q dot vector-area; no quadrature convention is reused.
    np.testing.assert_allclose(phi, space.topology.normals @ constant, atol=1.e-12)
    for c in range(mesh.n_cells):
        x = mesh.get_cell_centroid(c)[None, :]
        local = space.local_flux_values(c, phi, global_values=True)
        np.testing.assert_allclose(space.evaluate_flux(c, local, x), constant[None, :], atol=1.e-12)


def _supervised_problem():
    x = np.linspace(0, 1, 17)[:, None]
    return SupervisedProblem(MLP(1, hidden_layers=(8,), seed=5), PDEData(x, np.sin(np.pi * x)), Domain.interval())


def test_resume_exact_adam_state_and_inference(tmp_path):
    full = _supervised_problem().solve(epochs=8, run_dir=str(tmp_path / "full"), checkpoint_every=2)
    first = _supervised_problem().solve(epochs=3, run_dir=str(tmp_path / "resume"), checkpoint_every=2)
    resumed = _supervised_problem().solve(epochs=8, resume_from=first.run_dir / "last.pt")
    for a, b in zip(full.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert len(resumed.history) == 8
    state = torch.load(resumed.run_dir / "last.pt", weights_only=False)
    assert state["epoch"] == 8
    assert json.loads((resumed.run_dir / "status.json").read_text())["status"] == "completed"
    loaded = NeuralSolution.load(resumed.run_dir / "last.pt")
    np.testing.assert_array_equal(loaded([[.3], [.7]]), resumed([[.3], [.7]]))
    assert len(list(resumed.run_dir.glob("metrics_*.jsonl"))) == 2
    with pytest.raises(FileExistsError):
        _supervised_problem().solve(epochs=2, run_dir=str(resumed.run_dir))


def test_resume_restores_pinn_sampling_and_lbfgs_history(tmp_path):
    def pinn(seed):
        return PINNProblem(Domain.interval(), MLP(1, hidden_layers=(8,), seed=3),
                           f=2., n_interior=12, n_boundary=8, seed=seed)
    full = pinn(2).solve(epochs=4, run_dir=tmp_path / "pinn_full")
    first = pinn(2).solve(epochs=2, run_dir=tmp_path / "pinn_first")
    restored_problem = pinn(999)
    resumed = restored_problem.solve(epochs=4, resume_from=first.run_dir / "last.pt")
    for a, b in zip(full.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    full_lbfgs = _supervised_problem().solve(epochs=4, optimizer="lbfgs", lr=1.,
                                            run_dir=tmp_path / "lbfgs_full", lbfgs_max_iter=3)
    first_lbfgs = _supervised_problem().solve(epochs=2, optimizer="lbfgs", lr=1.,
                                             run_dir=tmp_path / "lbfgs_first", lbfgs_max_iter=3)
    resumed_lbfgs = _supervised_problem().solve(epochs=4, optimizer="lbfgs", lr=1., lbfgs_max_iter=3,
                                               resume_from=first_lbfgs.run_dir / "last.pt")
    for a, b in zip(full_lbfgs.model.parameters(), resumed_lbfgs.model.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_direct_mlp_multifield_and_tensor_graph(tmp_path):
    model = MLP(2, 2, hidden_layers=(8,), bounds=[[.1, .3], [0., 1.]])
    assert model(torch.ones((3, 2))).shape == (3, 2)
    sol = NeuralSolution(model, Domain.rectangle())
    x = torch.ones((3, 2), requires_grad=True)
    sol.tensor(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    selected = DirichletBC([[0., 0.]], 0., component=1)
    assert torch.isfinite(selected.loss(model, device="cpu", dtype=torch.float32))
    # Separate residual channels couple two output fields; one torch target per channel.
    domain = Domain.interval()
    c = DirichletBC([[0.], [1.]], lambda x: torch.cat([x, 2 * x], dim=1), backend="torch")
    problem = PINNProblem(domain, MLP(1, 2, hidden_layers=(8,)),
                          residual=lambda x, u: torch.cat([derivative(u, x, 0, component=0) - 1,
                                                           derivative(u, x, 0, component=1) - 2], dim=1),
                          conditions=[c], n_interior=8)
    assert set(problem.loss_terms(device="cpu", dtype=torch.float32)) == {"residual", "boundary"}
    with pytest.raises(ValueError, match="batch-independent"):
        PINNProblem(domain, torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 1)))


def test_interruption_failure_and_resume_guards(tmp_path):
    def interrupt(trainer, row):
        if row["epoch"] == 2:
            raise KeyboardInterrupt()
    run = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        _supervised_problem().solve(epochs=5, callback=interrupt, run_dir=str(run))
    assert json.loads((run / "status.json").read_text())["status"] == "interrupted"
    assert torch.load(run / "interrupted.pt", weights_only=False)["epoch"] == 2
    sol = _supervised_problem().solve(epochs=5, resume_from=run / "interrupted.pt")
    assert len(sol.history) == 5
    with pytest.raises(ValueError, match="lr"):
        _supervised_problem().solve(epochs=6, lr=.02, resume_from=run / "last.pt")
    assert torch.load(run / "last.pt", weights_only=False)["epoch"] == 5
    problem = PINNProblem(Domain.interval(), MLP(1), residual=lambda x, u: u * float("nan"))
    with pytest.raises(FloatingPointError):
        problem.solve(epochs=1, run_dir=str(tmp_path / "failed"))
    assert json.loads((tmp_path / "failed" / "status.json").read_text())["status"] == "failed"
    assert (tmp_path / "failed" / "last.pt").exists()


def test_pinn_poisson_converges_and_lbfgs(tmp_path):
    # Manufactured solution is a quadratic; check independent held-out coordinates.
    domain = Domain.interval()
    sol = solve_pinn(domain, f=2., hidden_layers=(16, 16), n_interior=48, n_boundary=8,
                     epochs=250, lr=.01, run_dir=str(tmp_path / "poisson"), checkpoint_every=100)
    x = np.linspace(0, 1, 101)[:, None]
    rmse = np.sqrt(np.mean((sol(x) - x[:, 0] * (1 - x[:, 0])) ** 2))
    assert rmse < .015
    assert sol.history[-1]["residual"] < sol.history[0]["residual"] * .05
    refined = PINNProblem(domain, sol.model, f=2., n_interior=48, n_boundary=8).solve(
        epochs=5, optimizer="lbfgs", lr=1., run_dir=str(tmp_path / "lbfgs"))
    assert np.sqrt(np.mean((refined(x) - x[:, 0] * (1 - x[:, 0])) ** 2)) < rmse


def test_supervised_converges_on_fem_data_and_hybrid_loss(tmp_path):
    mesh = Mesh.rectangle(nx=3, ny=3)
    v = FESpace.lagrange(mesh, 1)
    exact = lambda p: 1 + p[0] + 2 * p[1]
    traditional = PoissonProblem(v, f=lambda p: 0., g=exact).solve()
    data = PDEData.from_fe(v, traditional)
    sol = solve_supervised(data, domain=mesh, hidden_layers=(), epochs=200, lr=.05,
                           run_dir=str(tmp_path / "femdata"))
    np.testing.assert_allclose(sol(mesh.nodes), [exact(p) for p in mesh.nodes], atol=.002)
    hybrid = PINNProblem(mesh, MLP(2), f=0., g=exact, data=data)
    terms = hybrid.loss_terms(device="cpu", dtype=torch.float32)
    assert set(terms) == {"residual", "boundary", "data"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_smoke(tmp_path):
    sol = _supervised_problem().solve(epochs=2, device="cuda", dtype="float32", run_dir=str(tmp_path / "cuda"))
    assert sol.device.type == "cuda"
    assert np.isfinite(sol([[.5]])).all()
    pinn = solve_pinn(Domain.interval(), f=2., hidden_layers=(8,), epochs=2, n_interior=8,
                      device="cuda", dtype="float32", run_dir=tmp_path / "cuda_pinn")
    assert np.isfinite(pinn.gradient([[.5]])).all()
