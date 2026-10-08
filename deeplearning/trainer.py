"""Persistent Adam/L-BFGS training with atomic checkpoints and explicit resume."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import traceback
from uuid import uuid4

import numpy as np
import torch

from .solution import NeuralSolution


@dataclass
class TrainingConfig:
    epochs: int = 2000
    lr: float = 1.e-3
    optimizer: str = "adam"
    device: str = "cpu"
    dtype: str = "float64"
    seed: int = 0
    checkpoint_every: int = 50
    log_every: int = 100
    lbfgs_max_iter: int = 20
    tolerance: float = 0.
    run_dir: str | None = None

    def __post_init__(self):
        if self.run_dir is not None:
            self.run_dir = os.fspath(self.run_dir)
        for name in ("epochs", "checkpoint_every", "log_every", "lbfgs_max_iter"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.optimizer not in ("adam", "lbfgs"):
            raise ValueError("optimizer must be adam or lbfgs")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not np.isfinite(self.lr) or self.lr <= 0 or not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("lr must be positive and tolerance nonnegative")


def _time():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Trainer:
    """Epochs is the total target, including previously completed epochs on resume.

    callback(trainer, metrics) executes after a completed optimizer step.
    Every attempt has its own metrics JSONL and log file; history and checkpoints persist.
    """

    def __init__(self, config=None, *, resume_from=None, callback=None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("Pass TrainingConfig or keyword options, not both")
        self.config = config if config is not None else TrainingConfig(**kwargs)
        if not isinstance(self.config, TrainingConfig):
            raise TypeError("config must be TrainingConfig")
        self.resume_from = None if resume_from is None else Path(resume_from).resolve()
        self.callback = callback
        self.epoch, self.history = 0, []
        self.problem, self.optimizer = None, None
        self.run_dir = None
        self._restored = False
        self.checkpoint_epoch = None

    def _directory(self):
        if self.config.run_dir is not None:
            path = Path(self.config.run_dir).resolve()
        elif self.resume_from is not None:
            path = self.resume_from.parent
        else:
            workspace = Path(__file__).resolve().parents[2]
            root = workspace if (workspace / "WORK_PROGRESS.md").exists() else Path.cwd()
            path = root / "logs" / "deeplearning" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
        if self.resume_from is None and path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Nonempty run_dir {path}; choose a new directory or resume_from")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _status(self, status, **extra):
        _atomic_json(self.run_dir / "status.json", {"status": status, "epoch": self.epoch,
                     "target_epochs": self.config.epochs, "updated_at": _time(),
                     "session": self.session, "checkpoint": str(self.run_dir / "last.pt"),
                     "checkpoint_epoch": self.checkpoint_epoch,
                     "resume": "Recreate the same PDE/model and pass resume_from=last.pt with a larger total epochs target.",
                     **extra})

    def save_checkpoint(self, name="last.pt"):
        if self.problem is None or self.optimizer is None:
            return
        state = {"format_version": 1, "epoch": self.epoch, "history": self.history,
                 "model": self.problem.network.state_dict(),
                 "model_spec": getattr(self.problem.network, "spec", None),
                 "model_class": type(self.problem.network).__qualname__,
                 "optimizer": self.optimizer.state_dict(), "config": asdict(self.config),
                 "problem_metadata": self.problem.metadata(), "problem": self.problem.state_dict(),
                 "random": {"python": random.getstate(), "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
        if Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("Checkpoint name must be a .pt filename inside run_dir")
        path = self.run_dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if name == "last.pt":
            self.checkpoint_epoch = self.epoch

    def _restore(self):
        state = torch.load(self.resume_from, map_location="cpu", weights_only=False)
        if state.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        if state["problem_metadata"] != self.problem.metadata():
            raise ValueError("Checkpoint problem/conditions/weights differ; recreate the original PDE")
        model = self.problem.network
        if state["model_class"] != type(model).__qualname__ or state["model_spec"] != getattr(model, "spec", None):
            raise ValueError("Checkpoint network architecture differs")
        for key in ("optimizer", "dtype", "lr", "lbfgs_max_iter"):
            if state["config"][key] != getattr(self.config, key):
                raise ValueError(f"Resume must keep {key}; change settings in a separate new training run")
        if self.config.epochs < state["epoch"]:
            raise ValueError("epochs is a total target and cannot be less than the checkpoint epoch")
        model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.problem.load_state_dict(state["problem"])
        self.epoch, self.history = state["epoch"], state["history"]
        rng = state["random"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"] is not None:
            if len(rng["cuda"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(rng["cuda"])

    def fit(self, problem):
        self.problem = problem
        self.epoch, self.history, self.optimizer = 0, [], None
        self._restored = False
        self.checkpoint_epoch = None
        self.run_dir = self._directory()
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        logger = logging.getLogger(f"MosicaFE.deeplearning.{self.session}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(self.run_dir / f"training_{self.session}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        metrics_path = self.run_dir / f"metrics_{self.session}.jsonl"
        try:
            self._status("initializing")
            _atomic_json(self.run_dir / f"config_{self.session}.json", {"config": asdict(self.config),
                         "resume_from": str(self.resume_from) if self.resume_from else None,
                         "problem": problem.metadata(), "torch": str(torch.__version__)})
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            device = torch.device(self.config.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            dtype = getattr(torch, self.config.dtype)
            problem.network.to(device=device, dtype=dtype)
            parameters = list(problem.network.parameters())
            if self.config.optimizer == "adam":
                self.optimizer = torch.optim.Adam(parameters, lr=self.config.lr)
            else:
                self.optimizer = torch.optim.LBFGS(parameters, lr=self.config.lr,
                                                  max_iter=self.config.lbfgs_max_iter,
                                                  line_search_fn="strong_wolfe")
            if self.resume_from is not None:
                self._restore()
                self._restored = True
                logger.info("Resumed %s at epoch %d", self.resume_from, self.epoch)
            else:
                self.epoch, self.history = 0, []
            # Initial checkpoint exists even if the process is killed before the first step.
            self.save_checkpoint()
            self._status("running")
            logger.info("Started %s on %s: target epochs=%d", type(problem).__name__, device, self.config.epochs)
            problem.network.train()

            def closure():
                self.optimizer.zero_grad(set_to_none=True)
                terms = problem.loss_terms(device=device, dtype=dtype)
                loss = sum(terms.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError("Nonfinite training gradient")
                return loss

            with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics:
                while self.epoch < self.config.epochs:
                    if self.config.optimizer == "adam":
                        closure()
                        self.optimizer.step()
                    else:
                        self.optimizer.step(closure)
                    self.epoch += 1
                    # Report post-step losses so logged values describe the saved model.
                    terms = problem.loss_terms(device=device, dtype=dtype)
                    row = {"epoch": self.epoch, **{k: float(v.detach().cpu()) for k, v in terms.items()}}
                    row["total"] = sum(row[k] for k in terms)
                    if not np.isfinite(row["total"]):
                        raise FloatingPointError("Nonfinite post-step loss")
                    self.history.append(row)
                    metrics.write(json.dumps({**row, "time": _time()}, allow_nan=False) + "\n")
                    metrics.flush()
                    if self.epoch % self.config.checkpoint_every == 0:
                        self.save_checkpoint()
                        self._status("running", metrics=row)
                    if self.epoch == 1 or self.epoch % self.config.log_every == 0:
                        logger.info("epoch=%d losses=%s", self.epoch, row)
                    if self.callback is not None:
                        self.callback(self, row.copy())
                    if self.config.tolerance > 0 and row["total"] <= self.config.tolerance:
                        logger.info("Loss tolerance reached")
                        break
            self.save_checkpoint()
            self._status("completed", metrics=self.history[-1] if self.history else None)
            problem.network.eval()
            result = NeuralSolution(problem.network, getattr(problem, "domain", None),
                                    history=self.history, run_dir=self.run_dir)
            problem.solution = result
            logger.info("Completed at epoch %d, checkpoint %s", self.epoch, self.run_dir / "last.pt")
            return result
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            logger.exception("%s at epoch %d", status, self.epoch)
            # A failed restore must never overwrite the known-good checkpoint.
            exit_checkpoint = None
            if self.optimizer is not None and (self.resume_from is None or self._restored):
                try:
                    name = "interrupted.pt" if status == "interrupted" else "failure.pt"
                    self.save_checkpoint(name)
                    exit_checkpoint = str(self.run_dir / name)
                except Exception:
                    logger.exception("Could not save exit checkpoint; previous atomic checkpoint remains")
            self._status(status, error=repr(exc), traceback=traceback.format_exc(), exit_checkpoint=exit_checkpoint)
            raise
        finally:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
