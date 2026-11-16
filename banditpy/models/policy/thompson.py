import math

import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.special import betainc, betaincinv, beta as Bfn

from .base import BasePolicy, ParameterGroup, ParameterSpec
from .beta_schedule import NoBeta

# Gauss-Legendre nodes/weights on [0, 1] for 'p_greater' (16 nodes: max abs
# error ~4e-4, median ~6e-6 against adaptive quadrature over a wide range of
# Beta parameters).
_GL_U, _GL_W = np.polynomial.legendre.leggauss(16)
_GL_U = (_GL_U + 1.0) / 2.0
_GL_W = _GL_W / 2.0


def p_greater(a0, b0, a1, b1):
    """P(theta0 > theta1) for independent theta0 ~ Beta(a0, b0), theta1 ~ Beta(a1, b1).

    Deterministic quadrature, so the likelihood is a smooth function of the
    parameters. The expectation is taken over the quantiles of the narrower
    belief and evaluates the CDF of the broader one, which keeps the integrand
    smooth when one belief is much sharper than the other.
    """
    v0 = a0 * b0 / ((a0 + b0) ** 2 * (a0 + b0 + 1.0))
    v1 = a1 * b1 / ((a1 + b1) ** 2 * (a1 + b1 + 1.0))
    if v1 <= v0:  # E_theta1[1 - F0(theta1)]
        p = 1.0 - float(_GL_W @ betainc(a0, b0, betaincinv(a1, b1, _GL_U)))
    else:  # E_theta0[F1(theta0)]
        p = float(_GL_W @ betainc(a1, b1, betaincinv(a0, b0, _GL_U)))
    return min(max(p, 1e-12), 1.0 - 1e-12)


class BaseThompson(BasePolicy):
    """
    Shared posterior mechanics for Thompson sampling.
    Subclasses differ only in learning-rate structure.
    """

    def __init__(self, n_sim: int = 500, use_analytic: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.n_sim = n_sim
        self.use_analytic = use_analytic

    # --- State lifecycle ---

    def reset(self):
        self.s = np.zeros(2)
        self.f = np.zeros(2)

    def forget(self):
        self.s *= self.params["tau"]
        self.f *= self.params["tau"]

    # --- Thompson logits ---

    def logits(self):
        p = self.params

        alpha = np.maximum(p["alpha0"] + self.s, 1e-6)
        beta = np.maximum(p["beta0"] + self.f, 1e-6)

        if self.use_analytic:
            # posterior mean → DecisionModel applies softmax(beta·logits)
            return alpha / (alpha + beta)

        samples = np.random.beta(
            alpha[:, None],
            beta[:, None],
            size=(2, self.n_sim),
        )
        return samples.mean(axis=1)

    # --- Diagnostics ---

    def posterior_trajectory(self, task):
        """Return posterior (alpha,beta,mean) trajectories."""
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


class ThompsonShared(BaseThompson):
    """
    Thompson sampling:
    Shared learning rates for chosen / unchosen outcomes.
    """

    class Params(ParameterGroup):
        alpha0 = ParameterSpec("alpha0", (1e-3, 20.0), description="Prior alpha")
        beta0 = ParameterSpec("beta0", (1e-3, 20.0), description="Prior beta")
        tau = ParameterSpec("tau", (0.1, 0.999), description="Decay factor")
        lr_chosen = ParameterSpec(
            "lr_chosen", (0.0, 1.0), description="LR (chosen arm)"
        )
        lr_unchosen = ParameterSpec(
            "lr_unchosen", (0.0, 1.0), description="LR (unchosen arm)"
        )

    params: Params

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice

        lr_c_pos = lr_c_neg = p["lr_chosen"]
        lr_u_pos = lr_u_neg = p["lr_unchosen"]

        if reward == 1:
            self.s[choice] += lr_c_pos
            self.f[other] += lr_u_neg
        else:
            self.f[choice] += lr_c_neg
            self.s[other] += lr_u_pos


class ThompsonSplit(BaseThompson):
    """
    Thompson sampling:
    Fully independent pos/neg learning rates for both arms.
    """

    class Params(ParameterGroup):
        alpha0 = ParameterSpec("alpha0", (1, 20.0), description="Prior alpha")
        beta0 = ParameterSpec("beta0", (1, 20.0), description="Prior beta")
        tau = ParameterSpec("tau", (0.1, 0.999), description="Decay factor")
        lr_c_pos = ParameterSpec(
            "lr_c_pos", (0.0, 1.0), description="LR (chosen arm, positive reward)"
        )
        lr_c_neg = ParameterSpec(
            "lr_c_neg", (0.0, 1.0), description="LR (chosen arm, no reward)"
        )
        lr_u_pos = ParameterSpec(
            "lr_u_pos", (0.0, 1.0), description="LR (unchosen arm, positive reward)"
        )
        lr_u_neg = ParameterSpec(
            "lr_u_neg", (0.0, 1.0), description="LR (unchosen arm, no reward)"
        )

    params: Params

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice

        if reward == 1:
            self.s[choice] += p["lr_c_pos"]
            self.f[other] += p["lr_u_neg"]
        else:
            self.f[choice] += p["lr_c_neg"]
            self.s[other] += p["lr_u_pos"]


class ThompsonSticky(BasePolicy):
    """
    Thompson sampling with exact choice probabilities, a port bias and a
    choice (perseverance) trace.

    Beliefs about each port's reward probability, discounted evidence s, f:
        forget every trial   s <- tau * s,  f <- tau * f       (both ports)
        reward on chosen     s[c] += lr_chosen,   f[u] += lr_unchosen
        miss on chosen       f[c] += lr_chosen,   s[u] += lr_unchosen
        belief               theta_i ~ Beta(alpha0 + s_i, beta0 + f_i)
    Thompson sampling picks the port whose sampled theta is larger, so
        P_TS(port 0) = P(theta0 > theta1)    (computed exactly, 'p_greater')
    The sharper the beliefs, the more deterministic the choice: the beliefs'
    own spread plays the role of the softmax temperature, so there is no beta
    ('NoBeta'). On top of it, as in 'Qlearn' (sticky outside the choice rule):
        h += alpha_h * (choice - h)          # 0 = port 0, 1 = port 1
        logit P(port 0) = logit P_TS + 2 * bias - 2 * sticky * (h - 0.5)
    so bias = sticky = 0 is pure Thompson sampling. s = f = 0 and h = 0.5 at
    each reset. The unchosen update treats the ports as anti-correlated (a
    reward on one is evidence against the other); lr_unchosen = 0 learns about
    the chosen port only.
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha0 = ParameterSpec("alpha0", (0.5, 100.0), default=1.0, description="Prior alpha (pseudo-rewards)")
        beta0 = ParameterSpec("beta0", (0.5, 100.0), default=1.0, description="Prior beta (pseudo-misses)")
        tau = ParameterSpec("tau", (0.0, 0.999), default=0.9, description="Retention of evidence per trial")
        lr_chosen = ParameterSpec("lr_chosen", (0.0, 10.0), default=1.0, description="Evidence added to the chosen port")
        lr_unchosen = ParameterSpec(
            "lr_unchosen", (0.0, 10.0), default=0.0, description="Counterfactual evidence added to the unchosen port"
        )
        bias = ParameterSpec(
            "bias", (-3.0, 3.0), default=0.0, description="Bias toward port 0 vs port 1 (half-logit)"
        )
        alpha_h = ParameterSpec(
            "alpha_h", (0.0, 1.0), default=0.0, description="Learning rate of the choice (perseverance) trace"
        )
        sticky = ParameterSpec(
            "sticky", (0.0, 20.0), default=0.0, description="Perseverance weight (repeat recent choices)"
        )

    params: Params

    def reset(self):
        self.s = [0.0, 0.0]
        self.f = [0.0, 0.0]
        self.h = 0.5

    def forget(self):
        tau = self.params["tau"]
        self.s = [tau * self.s[0], tau * self.s[1]]
        self.f = [tau * self.f[0], tau * self.f[1]]

    def p_thompson(self):
        """P(port 0) under pure Thompson sampling, from the current beliefs."""
        a0, b0 = self.params["alpha0"], self.params["beta0"]
        return p_greater(a0 + self.s[0], b0 + self.f[0], a0 + self.s[1], b0 + self.f[1])

    def logits(self):
        # Softmax with beta = 1 over (z0, z1) gives sigmoid(z0 - z1), so
        # half the log-odds on each side reproduces P_TS exactly.
        p = self.p_thompson()
        half = 0.5 * (math.log(p) - math.log1p(-p))
        x = half + self.params["bias"] - self.params["sticky"] * (self.h - 0.5)
        return (x, -x)

    def update(self, choice, reward):
        lr_c = self.params["lr_chosen"]
        lr_u = self.params["lr_unchosen"]
        other = 1 - choice
        if reward:
            self.s[choice] += lr_c
            self.f[other] += lr_u
        else:
            self.f[choice] += lr_c
            self.s[other] += lr_u
        self.h += self.params["alpha_h"] * (choice - self.h)
