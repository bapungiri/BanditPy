import numpy as np
from scipy.optimize import minimize
from banditpy.core import Bandit2Arm


class Thompson2Arm:
    """
    Two‑arm Thompson model with:
      Discounted success / failure evidence (s_i, f_i):
         s_i <- tau * s_i
         f_i <- tau * f_i
         (effective memory ≈ 1/(1 - tau))
      Arm‑specific learning rates (lr1, lr2):
         chosen arm i:
            if reward=1: s_i += lr_i
            else:        f_i += lr_i
      Posterior parameters:
         alpha_i = 1 + s_i
         beta_i  = 1 + f_i
    Session starts (is_session_start) reset s,f to 0 (so alpha,beta back to 1).

    Parameters fitted: tau, lr1, lr2
    """

    def __init__(
        self,
        task: Bandit2Arm,
        n_sim: int = 500,
        seed: int = None,
        use_analytic: bool = False,
    ):
        self.choices = np.array(task.choices) - 1
        self.rewards = task.rewards.astype(float)
        self.is_session_start = task.is_session_start.astype(bool)
        self.n_arms = task.n_ports
        assert self.n_arms == 2, "Implemented for 2 arms."
        self.n_sim = n_sim
        self.use_analytic = use_analytic
        self._master_rng = np.random.default_rng(seed)

        self.tau: float | None = None
        self.lr1: float | None = None
        self.lr2: float | None = None
        self.nll: float | None = None

    # ---------- Choice probability ----------
    def _choice_prob(self, alpha, beta, choice, rng):
        if self.use_analytic:
            # P( Beta(a_c,b_c) > Beta(a_o,b_o) )
            c = choice
            o = 1 - choice
            # Numerical integral (fast enough for 2 arms small calls)
            from scipy.special import beta as Bfn, betainc

            a1, b1 = alpha[c], beta[c]
            a2, b2 = alpha[o], beta[o]
            # Simple Gauss-Legendre quadrature
            xs, ws = np.polynomial.legendre.leggauss(40)
            x = 0.5 * (xs + 1)
            w = 0.5 * ws
            pdf1 = x ** (a1 - 1) * (1 - x) ** (b1 - 1) / Bfn(a1, b1)
            cdf2 = betainc(a2, b2, 0, x)
            p = np.sum(pdf1 * cdf2 * w)
            return float(np.clip(p, 1e-6, 1 - 1e-6))
        else:
            samples = rng.beta(
                alpha[:, None], beta[:, None], size=(self.n_arms, self.n_sim)
            )
            picked = np.argmax(samples, axis=0)
            p = (picked == choice).mean()
            return float(np.clip(p, 1e-6, 1 - 1e-6))

    # ---------- Negative log-likelihood ----------
    def _calculate_log_likelihood(self, params):
        tau, lr1, lr2 = params
        if not (0 < tau <= 1) or lr1 <= 0 or lr2 <= 0:
            return np.inf

        # Common random numbers for determinism
        rng = np.random.default_rng()

        s = np.zeros(self.n_arms)
        f = np.zeros(self.n_arms)
        kappa = np.array([lr1, lr2])
        nll = 0.0

        for choice, reward, start_flag in zip(
            self.choices, self.rewards, self.is_session_start
        ):
            if start_flag:
                s[:] = 0.0
                f[:] = 0.0

            alpha = 1 + s
            beta = 1 + f
            p_choose = self._choice_prob(alpha, beta, choice, rng)
            nll -= np.log(p_choose)

            # Forgetting (discount)
            s *= tau
            f *= tau

            # Update chosen arm
            if reward == 1.0:
                s[choice] += kappa[choice]
            else:
                f[choice] += kappa[choice]

        return nll

    # ---------- Fit ----------
    def fit(
        self,
        bounds_tau=(0.5, 0.999),
        bounds_lr1=(0.01, 5.0),
        bounds_lr2=(0.01, 5.0),
        n_starts=10,
    ):
        bounds = [bounds_tau, bounds_lr1, bounds_lr2]
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        best_val = np.inf
        best_x = None

        for _ in range(n_starts):
            x0 = lo + (hi - lo) * self._master_rng.random(3)
            res = minimize(
                self._calculate_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=400, ftol=1e-6),
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        self.tau, self.lr1, self.lr2 = best_x
        self.nll = best_val
        return dict(tau=self.tau, lr1=self.lr1, lr2=self.lr2, nll=self.nll)

    # ---------- Diagnostics ----------
    def inspect_smoothness(self, repeats=20):
        if self.tau is None:
            raise RuntimeError("Fit first.")
        x = np.array([self.tau, self.lr1, self.lr2])
        vals = [self._calculate_log_likelihood(x) for _ in range(repeats)]
        vals = np.array(vals)
        print(
            f"Mean NLL={vals.mean():.2f}  SD={vals.std():.3f}  CoefVar={vals.std()/vals.mean():.4f}"
        )

    def bic(self):
        if self.nll is None:
            return np.nan
        k = 3
        return k * np.log(len(self.choices)) + 2 * self.nll

    def print_params(self):
        print(
            f"tau={self.tau:.4f}  lr1={self.lr1:.4f}  lr2={self.lr2:.4f}  NLL={self.nll:.2f}"
        )

    # ---------- Posterior trajectory ----------
    def simulate_posteriors(self):
        if self.tau is None:
            raise RuntimeError("Fit first.")
        s = np.zeros(self.n_arms)
        f = np.zeros(self.n_arms)
        alpha_traj = []
        beta_traj = []
        for choice, reward, start_flag in zip(
            self.choices, self.rewards, self.is_session_start
        ):
            if start_flag:
                s[:] = 0.0
                f[:] = 0.0
            alpha_traj.append(1 + s)
            beta_traj.append(1 + f)
            s *= self.tau
            f *= self.tau
            if reward == 1.0:
                s[choice] += self.lr1 if choice == 0 else self.lr2
            else:
                f[choice] += self.lr1 if choice == 0 else self.lr2
        # final
        alpha_traj.append(1 + s)
        beta_traj.append(1 + f)
        A = np.vstack(alpha_traj)
        B = np.vstack(beta_traj)
        mean = A / (A + B)
        return A, B, mean
