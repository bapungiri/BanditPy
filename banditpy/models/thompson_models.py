import numpy as np
from scipy.optimize import minimize
from banditpy.core import Bandit2Arm


class Thompson2Arm:
    """
    Two-arm Thompson sampling model with discounted success/failure evidence and separate learning rates for chosen and unchosen arms.

    Model logic:
      - For each trial, maintain discounted counts of successes (s_i) and failures (f_i) for each arm i.
      - At each trial, apply exponential forgetting:
            s_i <- tau * s_i
            f_i <- tau * f_i
        where tau ∈ (0, 1) controls the memory horizon (higher tau = longer memory).
      - Update chosen and unchosen arms:
        * If reward == 1:
            - Chosen arm: s_chosen += lr_chosen
            - Unchosen arm: s_unchosen += lr_unchosen
        * If reward == 0:
            - Chosen arm: f_chosen += lr_chosen
            - Unchosen arm: f_unchosen += lr_unchosen
        (If lr_unchosen = 0, unchosen arm is not updated.)
      - Posterior Beta parameters for each arm:
            alpha_i = alpha0 + s_i
            beta_i  = beta0 + f_i
        where alpha0, beta0 > 0 are prior parameters.
      - At reset_bool (Example: session_start, window_start), reset s_i and f_i to zero for both arms.

    Parameters fitted:
      - alpha0: prior success count (>0)
      - beta0: prior failure count (>0)
      - lr_chosen: learning rate for chosen arm (>0)
      - lr_unchosen: learning rate for unchosen arm (≥0)
      - tau: forgetting factor (0 < tau < 1)

    Choice probability:
      - At each trial, compute P(choice) using either Monte Carlo Thompson sampling or analytic Beta comparison.

    Simulation:
      - The simulate_posteriors() method returns the trajectory of alpha/beta parameters and posterior means for each arm over trials.

    Effective memory horizon:
      - Approximately 1/(1 - tau)

    Notes:
      - Setting lr_unchosen = 0 disables fictive updates for the unchosen arm.
      - All parameters are constrained to valid ranges during fitting.
    """

    def __init__(
        self,
        task: Bandit2Arm,
        reset_bool: bool = None,  # Flags for resetting s and f, default session_start
        n_sim: int = 500,
        seed: int = None,
        use_analytic: bool = False,
    ):
        self.choices = np.array(task.choices) - 1  # 0-based
        self.rewards = task.rewards.astype(float)

        if reset_bool is None:
            self.reset_bool = task.is_session_start.astype(bool)
        else:
            self.reset_bool = np.array(reset_bool).astype(bool)

        self.n_arms = task.n_ports
        assert self.n_arms == 2, "Implemented for 2 arms."
        self.n_sim = n_sim
        self.use_analytic = use_analytic
        self._master_rng = np.random.default_rng(seed)

        self.a0: float | None = None
        self.b0: float | None = None
        self.lr_chosen: float | None = None
        self.lr_unchosen: float | None = None
        self.tau: float | None = None
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
        alpha0, beta0, lr_chosen, lr_unchosen, tau = params

        # Common random numbers for determinism
        rng = np.random.default_rng()

        s = np.zeros(self.n_arms)
        f = np.zeros(self.n_arms)
        nll = 0.0

        for choice, reward, reset in zip(self.choices, self.rewards, self.reset_bool):
            if reset:
                s[:] = 0.0
                f[:] = 0.0

            alpha = np.maximum(alpha0 + s, alpha0)
            beta = np.maximum(beta0 + f, beta0)
            p_choose = self._choice_prob(alpha, beta, choice, rng)
            nll -= np.log(p_choose)

            # Forgetting (discount)
            s *= tau
            f *= tau

            # Update chosen and unchosen arms
            if reward == 1.0:
                s[choice] += lr_chosen  # update success of chosen
                f[1 - choice] += lr_unchosen  # update failure of unchosen
            else:
                f[choice] += lr_chosen  # update failure of chosen
                s[1 - choice] += lr_unchosen  # update success of unchosen

        return nll

    # ---------- Fit ----------
    def fit(
        self,
        alpha0=(1, 10),
        beta0=(1, 10),
        lr_chosen=(0.01, 1),
        lr_unchosen=(0.01, 1),
        tau=(0.1, 1),
        n_starts=10,
    ):
        bounds = [alpha0, beta0, lr_chosen, lr_unchosen, tau]
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        best_val = np.inf
        best_x = None

        for _ in range(n_starts):
            x0 = lo + (hi - lo) * self._master_rng.random(5)
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

        self.alpha0, self.beta0, self.lr_chosen, self.lr_unchosen, self.tau = best_x
        self.nll = best_val
        return dict(
            alpha0=self.alpha0,
            beta0=self.beta0,
            lr_chosen=self.lr_chosen,
            lr_unchosen=self.lr_unchosen,
            tau=self.tau,
            nll=self.nll,
        )

    # ---------- Diagnostics ----------
    def inspect_smoothness(self, repeats=20):
        if self.tau is None:
            raise RuntimeError("Fit first.")
        x = np.array(
            [self.alpha0, self.beta0, self.lr_chosen, self.lr_unchosen, self.tau]
        )
        vals = [self._calculate_log_likelihood(x) for _ in range(repeats)]
        vals = np.array(vals)
        print(
            f"Mean NLL={vals.mean():.2f}  SD={vals.std():.3f}  CoefVar={vals.std()/vals.mean():.4f}"
        )

    def bic(self):
        if self.nll is None:
            return np.nan
        k = 5
        return k * np.log(len(self.choices)) + 2 * self.nll

    def print_params(self):
        print(
            f"alpha0={self.alpha0:.4f}, beta0={self.beta0:.4f}, lr_chosen={self.lr_chosen:.4f},  lr_unchosen={self.lr_unchosen:.4f}, tau={self.tau:.4f}, NLL={self.nll:.2f}"
        )

    # ---------- simulate posterior ----------
    def simulate_posteriors(self):
        if any(
            p is None
            for p in (
                self.alpha0,
                self.beta0,
                self.lr_chosen,
                self.lr_unchosen,
                self.tau,
            )
        ):
            raise RuntimeError("Call fit() first.")
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
            alpha_traj.append(self.alpha0 + s)
            beta_traj.append(self.beta0 + f)
            s *= self.tau
            f *= self.tau
            if reward == 1.0:
                s[choice] += self.lr_chosen
                if self.lr_unchosen > 0:
                    s[1 - choice] += self.lr_unchosen
            else:
                f[choice] += self.lr_chosen
                if self.lr_unchosen > 0:
                    f[1 - choice] += self.lr_unchosen
        alpha_traj.append(self.alpha0 + s)
        beta_traj.append(self.beta0 + f)
        A = np.vstack(alpha_traj)
        B = np.vstack(beta_traj)
        mean = A / (A + B)
        return A, B, mean
