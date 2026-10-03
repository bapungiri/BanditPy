import numpy as np
from scipy.optimize import minimize, differential_evolution
from joblib import Parallel, delayed
from tqdm import tqdm


class BaseOptimizer:
    """Protocol for optimizers used by DecisionModel.

    Subclasses must implement fit(objective, bounds, seeds, n_jobs, progress)
    and return (best_fun, best_x, fvals).
    """

    def fit(
        self, objective, bounds, seeds, n_jobs=1, progress=False
    ):  # pragma: no cover - interface
        raise NotImplementedError


def _fun_and_grad(objective, lo, hi):
    """Objective plus a forward-difference gradient, computed as scipy's
    default '2-point' scheme does (same step, flipped at an upper bound),
    so fits are unchanged. Supplying it ourselves lets us clip x to the
    bounds first: scipy 1.15's L-BFGS-B occasionally hands back an x a
    hair outside them, and its own gradient code then raises "'x0'
    violates bound constraints", killing the whole multi-start fit.
    """
    rel_step = np.sqrt(np.finfo(float).eps)

    def fg(x):
        x = np.clip(x, lo, hi)
        f = objective(x)
        g = np.empty_like(x)
        for i in range(x.size):
            h = rel_step * (1.0 if x[i] >= 0 else -1.0) * max(1.0, abs(x[i]))
            if x[i] + h > hi[i] or x[i] + h < lo[i]:
                h = -h
            xh = x.copy()
            xh[i] = x[i] + h
            g[i] = (objective(xh) - f) / (xh[i] - x[i])
        return f, g

    return fg


class LBFGSOptimizer(BaseOptimizer):
    def fit(self, objective, bounds, seeds, n_jobs=1, progress=False):
        lo_hi = [b for _, b in bounds]
        lo, hi = np.array(lo_hi, dtype=float).T
        fg = _fun_and_grad(objective, lo, hi)

        def _run(seed):
            rng = np.random.default_rng(seed)
            x0 = np.array([rng.uniform(*b) for b in lo_hi])
            try:
                res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=lo_hi)
            except Exception as err:  # one bad start shouldn't sink the others
                print(f"L-BFGS-B start {seed} failed: {err!r}")
                return np.nan, x0
            return res.fun, np.clip(res.x, lo, hi)

        iterator = seeds
        if progress and n_jobs == 1:
            iterator = tqdm(iterator, desc="LBFGS starts")

        if n_jobs == 1:
            results = [_run(s) for s in iterator]
        else:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run)(s) for s in iterator
            )

        fvals = np.array([r[0] for r in results], dtype=float)
        if np.isnan(fvals).all():
            raise RuntimeError("All L-BFGS-B starts failed")
        best = int(np.nanargmin(fvals))
        return fvals[best], results[best][1], fvals


class DEOptimizer(BaseOptimizer):
    def __init__(self, popsize=15, maxiter=200, tol=1e-6):
        self.popsize = popsize
        self.maxiter = maxiter
        self.tol = tol

    def fit(self, objective, bounds, seeds, n_jobs=1, progress=False):
        lo_hi = [b for _, b in bounds]

        def _run(seed):
            res = differential_evolution(
                objective,
                bounds=lo_hi,
                maxiter=self.maxiter,
                popsize=self.popsize,
                tol=self.tol,
                seed=int(seed),
                polish=True,
            )
            return res.fun, res.x

        iterator = seeds
        if progress and n_jobs == 1:
            iterator = tqdm(iterator, desc="DE starts")

        if n_jobs == 1:
            results = [_run(s) for s in iterator]
        else:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run)(s) for s in iterator
            )

        best_fun, best_x = min(results, key=lambda t: t[0])
        fvals = np.array([r[0] for r in results], dtype=float)
        return best_fun, best_x, fvals


class OptunaOptimizer(BaseOptimizer):
    """Multi-start Optuna search, one independent study per seed.

    'log_params' names parameters sampled on a log scale (e.g. "beta");
    their lower bound must be > 0. Names not being fitted are ignored.
    """

    def __init__(
        self,
        n_trials=100,
        timeout=None,
        sampler=None,
        pruner=None,
        show_progress=False,
        log_params=(),
    ):
        self.log_params = frozenset(log_params)
        self.n_trials = n_trials
        self.timeout = timeout
        self.sampler = sampler
        self.pruner = pruner
        self.show_progress = show_progress

    def fit(self, objective, bounds, seeds, n_jobs=1, progress=False):
        import optuna

        # Silence Optuna logs in worker processes.
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        optuna.logging.disable_default_handler()

        def _make_sampler(seed):
            if self.sampler is None:
                return optuna.samplers.TPESampler(seed=int(seed))
            if isinstance(self.sampler, str):
                name = self.sampler.lower()
                if name == "tpe":
                    return optuna.samplers.TPESampler(seed=int(seed))
                if name == "cma":
                    return optuna.samplers.CmaEsSampler(seed=int(seed))
                raise ValueError(f"Unknown Optuna sampler '{self.sampler}'")
            return self.sampler

        def _make_pruner():
            if self.pruner is None:
                return optuna.pruners.NopPruner()
            if isinstance(self.pruner, str):
                name = self.pruner.lower()
                if name == "nop":
                    return optuna.pruners.NopPruner()
                if name == "median":
                    return optuna.pruners.MedianPruner()
                return optuna.pruners.NopPruner()
            return self.pruner

        for name, (low, _) in bounds:
            if name in self.log_params and low <= 0:
                raise ValueError(
                    f"log-scale parameter '{name}' needs a lower bound > 0, got {low}"
                )

        def _run(seed):
            sampler = _make_sampler(seed)
            pruner = _make_pruner()

            def _objective(trial):
                theta = [
                    trial.suggest_float(name, low, high, log=name in self.log_params)
                    for name, (low, high) in bounds
                ]
                return objective(theta)

            study = optuna.create_study(
                direction="minimize", sampler=sampler, pruner=pruner
            )
            show_bar = self.show_progress and n_jobs == 1 and len(seeds) == 1
            study.optimize(
                _objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=1,
                show_progress_bar=show_bar,
            )

            best_trial = study.best_trial
            best_fun = float(best_trial.value)
            best_x = np.array(
                [best_trial.params[name] for name, _ in bounds], dtype=float
            )
            return best_fun, best_x

        iterator = seeds
        if progress and n_jobs == 1:
            iterator = tqdm(iterator, desc="Optuna starts")

        if n_jobs == 1:
            results = [_run(s) for s in iterator]
        else:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run)(s) for s in iterator
            )

        best_fun, best_x = min(results, key=lambda t: t[0])
        fvals = np.array([r[0] for r in results], dtype=float)
        return best_fun, best_x, fvals


def resolve_optimizer(optimizer=None):
    """Factory to resolve optimizer input (None, string, or BaseOptimizer)."""
    if optimizer is None or (
        isinstance(optimizer, str) and optimizer.lower() == "lbfgs"
    ):
        return LBFGSOptimizer()

    if isinstance(optimizer, str):
        name = optimizer.lower()
        if name == "de":
            return DEOptimizer()
        if name == "optuna":
            return OptunaOptimizer()
        raise ValueError(f"Unknown optimizer '{optimizer}'")

    if isinstance(optimizer, BaseOptimizer):
        return optimizer

    raise TypeError("optimizer must be None, a string, or a BaseOptimizer instance")
