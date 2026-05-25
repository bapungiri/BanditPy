import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


@dataclass
class ParameterSpec:
    name: str
    bounds: Tuple[float, float]
    default: Optional[float] = None
    active: bool = True
    description: str = ""

    def copy(self):
        return ParameterSpec(
            name=self.name,
            bounds=tuple(self.bounds),
            default=self.default,
            active=self.active,
            description=self.description,
        )


class BoundsRegistry:
    """
    Dict-style bounds editor.
    Safe for pickling and parallel execution.

    Examples
    --------
    policy.bounds["alpha_c"] = (0.0, 0.5)   # tighten learning-rate search range
    policy.bounds.enable("tau")
    policy.bounds.disable("lr_unchosen")
    """

    def __init__(self, specs: Dict[str, ParameterSpec]):
        self._specs = specs

    # --- dict-like access ---

    def __getitem__(self, name):
        if name not in self._specs:
            raise KeyError(f"No such parameter '{name}'")
        return self._specs[name].bounds

    def __setitem__(self, name, value):
        if name not in self._specs:
            raise KeyError(f"No such parameter '{name}'")
        self._specs[name].bounds = tuple(value)

    # --- enable / disable parameters for fitting ---

    def enable(self, name):
        self._specs[name].active = True

    def disable(self, name):
        self._specs[name].active = False

    # --- helpers ---

    def as_list(self, names: List[str]):
        return [self._specs[n].bounds for n in names]

    def to_dict(self):
        return {k: p.bounds for k, p in self._specs.items()}


class BasePolicy:
    """
    Subclasses define ``parameters = [ParameterSpec(...), ...]``.

    Each policy owns its ``beta_schedule`` (a ``BetaSchedule`` instance that
    controls the softmax inverse temperature used by ``DecisionModel``). The
    default is ``StaticBeta()``; override via the ``beta_schedule`` constructor
    argument or by setting ``default_beta_schedule`` as a class variable.

    Lifecycle:
        __init__()  → reset()  (policy starts in a valid state)
        reset()     → initialize state for a new session
        forget()    → optional decay step between trials
        logits()    → compute choice values
        update()    → apply learning update
    """

    common_parameters: List[ParameterSpec] = []

    parameters: List[ParameterSpec] = []
    _disable_common: List[str] = []
    default_beta_schedule = None  # BetaSchedule subclass; None → StaticBeta

    def __init__(self, beta_schedule=None, **kwargs):
        all_params = self.parameters + self.common_parameters
        self._param_specs: Dict[str, ParameterSpec] = {
            p.name: p.copy() for p in all_params
        }

        # Disable common parameters declared by subclass
        for name in self._disable_common:
            spec = self._param_specs[name]
            spec.active = False
            if spec.default is None:
                spec.default = 0.0

        # Runtime parameter values after fit() or set_params()
        self.params: Dict[str, float] = {}

        # Dict-style bounds API
        self.bounds = BoundsRegistry(self._param_specs)

        # Resolve beta_schedule: explicit arg > subclass default > StaticBeta.
        # StaticBeta is imported lazily to avoid a circular import
        # (beta_schedule.py already imports from base.py).
        if beta_schedule is None:
            from .beta_schedule import StaticBeta  # noqa: PLC0415

            beta_schedule = (self.default_beta_schedule or StaticBeta)()
        self.beta_schedule = beta_schedule

    # ---------------- Parameter API ----------------

    def param_names(self):
        return list(self._param_specs.keys())

    def active_parameter_names(self):
        own = [k for k, p in self._param_specs.items() if p.active]
        return own + self.beta_schedule.active_parameter_names()

    def get_bounds(self):
        d = self.bounds.to_dict()
        d.update(self.beta_schedule.get_bounds())
        return d

    def set_bounds(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._param_specs:
                raise ValueError(f"Unknown parameter '{k}'")
            self._param_specs[k].bounds = tuple(v)

    def set_params(self, params: Dict[str, float]):
        beta_names = set(self.beta_schedule.parameter_specs())
        sched_params = {}
        for k, v in params.items():
            if k in beta_names:
                sched_params[k] = v
            elif k in self._param_specs:
                self.params[k] = float(v)
            else:
                raise ValueError(f"Unknown parameter '{k}'")
        if sched_params:
            self.beta_schedule.set_params(sched_params)

    def get_param_defaults(self):
        return {k: p.default for k, p in self._param_specs.items()}

    def parameter_specs(self):
        combined = dict(self._param_specs)
        combined.update(self.beta_schedule.parameter_specs())
        return combined

    def describe(self, as_markdown: bool = False):
        """
        Print (and optionally return) a table of parameter metadata.

        Shows policy parameters and beta schedule parameters in separate sections.

        Parameters
        ----------
        as_markdown : bool, optional
            If True, returns a Markdown table string instead of printing.
        """
        headers = ["Parameter", "Bounds", "Default", "Active", "Description"]

        def _make_rows(specs):
            rows = []
            for p in specs.values():
                default_str = "—" if p.default is None else f"{p.default}"
                rows.append(
                    [
                        p.name,
                        f"{p.bounds}",
                        default_str,
                        "True" if p.active else "False",
                        p.description or "",
                    ]
                )
            return rows

        sched_specs = self.beta_schedule.parameter_specs()
        sched_name = self.beta_schedule.__class__.__name__

        if not as_markdown:
            col_hdr = (
                f"{headers[0]:<14}{headers[1]:<18}{headers[2]:<8}"
                f"{headers[3]:<8}{headers[4]}"
            )
            sep = "-" * 90
            print("Policy parameters")
            print(sep)
            print(col_hdr)
            print(sep)
            for r in _make_rows(self._param_specs):
                print(f"{r[0]:<14}{r[1]:<18}{r[2]:<8}{r[3]:<8}{r[4]}")
            print(f"\nBeta schedule ({sched_name})")
            print(sep)
            if sched_specs:
                print(col_hdr)
                print(sep)
                for r in _make_rows(sched_specs):
                    print(f"{r[0]:<14}{r[1]:<18}{r[2]:<8}{r[3]:<8}{r[4]}")
            else:
                print("  (no parameters)")
            return

        # ---- Markdown export ----
        md = ["### Policy parameters\n"]
        md += [
            "| Parameter | Bounds | Default | Active | Description |",
            "|----------:|--------|--------:|:------:|-------------|",
        ]
        for r in _make_rows(self._param_specs):
            md.append(f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} |")
        md.append(f"\n### Beta schedule ({sched_name})\n")
        if sched_specs:
            md += [
                "| Parameter | Bounds | Default | Active | Description |",
                "|----------:|--------|--------:|:------:|-------------|",
            ]
            for r in _make_rows(sched_specs):
                md.append(f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} |")
        else:
            md.append("*(no parameters)*")
        return "\n".join(md)

    # ---------------- Lifecycle hooks ----------------

    def reset(self):
        """Initialize internal state (must be implemented by subclasses)."""
        raise NotImplementedError

    def forget(self):
        """Optional decay step; default is a no-op."""
        pass

    def logits(self):
        raise NotImplementedError

    def update(self, choice, reward):
        raise NotImplementedError
