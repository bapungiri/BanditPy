import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.special import betainc, beta as Bfn

from .base import BasePolicy, ParameterSpec


class Thompson2Arm(BasePolicy):
    """
    Thompson sampling with discounted evidence and counterfactual updates.
    Supports learning-rate tying via lr_mode.
    """

    def __init__(
        self,
        lr_mode: str = "shared",  # shared | chosen_split | full_split
        n_sim: int = 500,
        use_analytic: bool = False,
        prior_strength: float = 1.0,
    ):
        self.lr_mode = lr_mode
        self.n_sim = n_sim
        self.use_analytic = use_analytic
        self.prior_strength = prior_strength

        assert lr_mode in {"shared", "chosen_split", "full_split"}

    @property
    def parameters(self) -> List[ParameterSpec]:
        params = [
            ParameterSpec("alpha0", (1e-3, 20.0)),
            ParameterSpec("beta0", (1e-3, 20.0)),
            ParameterSpec("tau", (0.1, 0.999)),
        ]

        if self.lr_mode == "shared":
            params += [
                ParameterSpec("lr_chosen", (0.0, 1.0)),
                ParameterSpec("lr_unchosen", (0.0, 1.0)),
            ]

        elif self.lr_mode == "chosen_split":
            params += [
                ParameterSpec("lr_c_pos", (0.0, 1.0)),
                ParameterSpec("lr_c_neg", (0.0, 1.0)),
                ParameterSpec("lr_unchosen", (0.0, 1.0)),
            ]

        else:  # full_split
            params += [
                ParameterSpec("lr_c_pos", (0.0, 1.0)),
                ParameterSpec("lr_c_neg", (0.0, 1.0)),
                ParameterSpec("lr_u_pos", (0.0, 1.0)),
                ParameterSpec("lr_u_neg", (0.0, 1.0)),
            ]

        # Needed for likelihood (softmax on Thompson samples)
        params.append(ParameterSpec("beta", (0.1, 10.0)))

        return params

    def reset(self):
        self.s = np.zeros(2)
        self.f = np.zeros(2)

    def forget(self):
        self.s *= self.params["tau"]
        self.f *= self.params["tau"]

    def logits(self):
        alpha = np.maximum(self.params["alpha0"] + self.s, 1e-6)
        beta = np.maximum(self.params["beta0"] + self.f, 1e-6)

        if self.use_analytic:
            # return posterior means (DecisionModel softmaxes them)
            return alpha / (alpha + beta)

        # Monte Carlo Thompson samples
        samples = np.random.beta(
            alpha[:, None],
            beta[:, None],
            size=(2, self.n_sim),
        )
        return samples.mean(axis=1)

    def update(self, choice, reward):
        p = self.params

        # Expand learning rates depending on tying
        if self.lr_mode == "shared":
            lr_c_pos = lr_c_neg = p["lr_chosen"]
            lr_u_pos = lr_u_neg = p["lr_unchosen"]

        elif self.lr_mode == "chosen_split":
            lr_c_pos = p["lr_c_pos"]
            lr_c_neg = p["lr_c_neg"]
            lr_u_pos = lr_u_neg = p["lr_unchosen"]

        else:  # full_split
            lr_c_pos = p["lr_c_pos"]
            lr_c_neg = p["lr_c_neg"]
            lr_u_pos = p["lr_u_pos"]
            lr_u_neg = p["lr_u_neg"]

        other = 1 - choice

        if reward == 1:
            self.s[choice] += lr_c_pos
            self.f[other] += lr_u_neg
        else:
            self.f[choice] += lr_c_neg
            self.s[other] += lr_u_pos

    # --------------------------------------------------
    # Posterior trajectories (for diagnostics)
    # --------------------------------------------------

    def posterior_trajectory(self, task):
        self.reset()
        A, B = [], []

        for c, r, reset in zip(
            task.choices - 1,
            task.rewards,
            task.is_session_start,
        ):
            if reset:
                self.reset()

            A.append(self.params["alpha0"] + self.s)
            B.append(self.params["beta0"] + self.f)

            self.forget()
            self.update(c, r)

        A = np.vstack(A)
        B = np.vstack(B)
        mean = A / (A + B)
        return A, B, mean
