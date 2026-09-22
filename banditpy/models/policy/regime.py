import numpy as np
from .base import BasePolicy, ParameterGroup, ParameterSpec
from .beta_schedule import NoBeta

N_REGIMES = 3


def _softmax(x: np.ndarray, beta: float) -> np.ndarray:
    z = beta * x
    z -= z.max()
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        return np.full_like(x, 1.0 / len(x))
    return e / s


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


class Qlearn3Regime(BasePolicy):
    """
    3-regime HMM: three independent, symmetric 'Qlearn'-style agents
    ('q1'/'q2'/'q3', own 'alpha_c_k'/'alpha_u_k'), updated unconditionally
    every trial — NOT frozen when a regime isn't responsible, unlike the
    original regime-owned design still used by 'QlearnDiff3StayRegime'.
    Regime k reads out ONLY its own dedicated agent k through a single
    inverse temperature 'beta_k' — no cross-agent mixing (contrast with
    'Qlearn2Regime'/'MoARegime', which weight ALL agents into every
    regime's value via a full weight matrix; this is the simpler
    one-agent-per-regime alternative to that).

        p(y | regime k) = softmax(q_{k+1}(y), beta_k)

    Per-regime persistence 'stay_0'/'stay_1'/'stay_2' (off-diagonal mass
    split uniformly between the other two regimes, same design as
    'QlearnDiff3StayRegime') gives a closed-form stationary occupancy,
    'pi_k = 1/(1-stay_k)' normalized — see 'occupancy()'. The initial
    belief at 'reset()' is a hardcoded uniform prior (no free parameter);
    this is the only thing the belief has to go on before any evidence
    arrives, so occupancy/'stay_k' entirely determine its long-run
    behavior within a session/window.

    'bias' is a simple additive shift toward arm 0 (away from arm 1),
    SHARED across all three regimes — a persistent port-side preference
    independent of which regime is currently responsible, rather than a
    regime-specific quantity:

        V_k(y=0) = beta_k * q_{k+1}(0) + bias
        V_k(y=1) = beta_k * q_{k+1}(1) - bias
        p(y | regime k) = softmax(V_k)

    'logits()' returns the log of the belief-weighted mixture of each
    regime's choice probs (paired with 'NoBeta'). Because the belief
    update is standard HMM forward-filtering, per-trial log 'logits()'
    summed by 'DecisionModel' is exactly the HMM marginal log-likelihood,
    so this fits directly into 'fit()'/'cross_validate()', no EM needed.

    Per trial: lik[k] = P(choice | regime k) from pre-update agent
    values; resp = normalize(b * lik); q1, q2, q3 updated unconditionally
    (NOT scaled by resp) via pe = reward - q[c], q[c] += alpha_c*pe,
    q[~c] += alpha_u*pe; b <- resp @ T(stay_0, stay_1, stay_2).
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_c_1 = ParameterSpec(
            "alpha_c_1", (0.0, 0.99), description="Learning rate (chosen), agent 1"
        )
        alpha_u_1 = ParameterSpec(
            "alpha_u_1", (-0.99, 0.99), description="Learning rate (unchosen), agent 1"
        )
        alpha_c_2 = ParameterSpec(
            "alpha_c_2", (0.0, 0.99), description="Learning rate (chosen), agent 2"
        )
        alpha_u_2 = ParameterSpec(
            "alpha_u_2", (-0.99, 0.99), description="Learning rate (unchosen), agent 2"
        )
        alpha_c_3 = ParameterSpec(
            "alpha_c_3", (0.0, 0.99), description="Learning rate (chosen), agent 3"
        )
        alpha_u_3 = ParameterSpec(
            "alpha_u_3", (-0.99, 0.99), description="Learning rate (unchosen), agent 3"
        )
        beta_0 = ParameterSpec(
            "beta_0", (0.1, 20.0), description="Inverse temp, regime 0 (agent 1)"
        )
        beta_1 = ParameterSpec(
            "beta_1", (0.1, 20.0), description="Inverse temp, regime 1 (agent 2)"
        )
        beta_2 = ParameterSpec(
            "beta_2", (0.1, 20.0), description="Inverse temp, regime 2 (agent 3)"
        )
        bias = ParameterSpec(
            "bias",
            (-10.0, 10.0),
            description="Additive shift toward arm 0 (shared across regimes)",
        )
        stay_0 = ParameterSpec(
            "stay_0", (0.0, 0.99), description="Probability of remaining in regime 0"
        )
        stay_1 = ParameterSpec(
            "stay_1", (0.0, 0.99), description="Probability of remaining in regime 1"
        )
        stay_2 = ParameterSpec(
            "stay_2", (0.0, 0.99), description="Probability of remaining in regime 2"
        )

    params: Params

    def reset(self):
        self.q1 = np.full(2, 0.5)
        self.q2 = np.full(2, 0.5)
        self.q3 = np.full(2, 0.5)
        self.b = np.full(N_REGIMES, 1.0 / N_REGIMES)

    def forget(self):
        pass

    def get_state(self):
        return self.b.copy()

    def _agent_values(self):
        return np.vstack([self.q1, self.q2, self.q3])

    def _regime_choice_probs(self):
        agents = self._agent_values()  # (N_REGIMES, 2)
        p = self.params
        betas = (p["beta_0"], p["beta_1"], p["beta_2"])
        V = np.vstack([betas[k] * agents[k] for k in range(N_REGIMES)])
        V[:, 0] += p["bias"]
        V[:, 1] -= p["bias"]
        return np.vstack([_softmax(V[k], 1.0) for k in range(N_REGIMES)])

    def logits(self):
        opt_probs = self._regime_choice_probs()
        p_action = self.b @ opt_probs
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def _transition_matrix(self):
        p = self.params
        stays = (p["stay_0"], p["stay_1"], p["stay_2"])
        T = np.empty((N_REGIMES, N_REGIMES))
        for k in range(N_REGIMES):
            switch_k = (1.0 - stays[k]) / (N_REGIMES - 1)
            T[k, :] = switch_k
            T[k, k] = stays[k]
        return T

    def update(self, choice, reward):
        opt_probs = self._regime_choice_probs()
        lik = opt_probs[:, choice]

        resp = self.b * lik
        resp_sum = resp.sum()
        if resp_sum <= 0:
            resp = np.full(N_REGIMES, 1.0 / N_REGIMES)
        else:
            resp /= resp_sum

        p = self.params
        other = 1 - choice

        # unconditional agent updates — not scaled by resp
        pe1 = reward - self.q1[choice]
        self.q1[choice] += p["alpha_c_1"] * pe1
        self.q1[other] += p["alpha_u_1"] * pe1
        np.clip(self.q1, 0.0, 1.0, out=self.q1)

        pe2 = reward - self.q2[choice]
        self.q2[choice] += p["alpha_c_2"] * pe2
        self.q2[other] += p["alpha_u_2"] * pe2
        np.clip(self.q2, 0.0, 1.0, out=self.q2)

        pe3 = reward - self.q3[choice]
        self.q3[choice] += p["alpha_c_3"] * pe3
        self.q3[other] += p["alpha_u_3"] * pe3
        np.clip(self.q3, 0.0, 1.0, out=self.q3)

        self.b = resp @ self._transition_matrix()
        self.b /= self.b.sum()

    def occupancy(self):
        """Closed-form stationary occupancy, 'pi_k = 1/(1-stay_k)'
        normalized — long-run fraction of trials in each regime."""
        p = self.params
        inv_leave = np.array(
            [
                1.0 / (1.0 - p["stay_0"]),
                1.0 / (1.0 - p["stay_1"]),
                1.0 / (1.0 - p["stay_2"]),
            ]
        )
        return inv_leave / inv_leave.sum()


class QlearnDiff1StayRegime(BasePolicy):
    """
    'Qlearn3Regime' with 'QlearnDiff's difference-based update in place of
    separate chosen/unchosen rates: one 'alpha_k' per regime, still moving
    both options each trial (7 free params vs. 10). With 'stay'->1, a
    single regime reduces to 'QlearnDiff'.

    Note: shared 'stay' forces occupancy to be exactly uniform (1/3 each)
    — see 'QlearnDiff3StayRegime' for per-regime 'stay'.

    Per trial: lik[k] = P(choice | regime k); resp = normalize(b * lik);
    diff_k = q[k,c] - q[k,~c]; rpe_k = reward - abs(diff_k);
    delta_k = alpha_k * resp[k] * rpe_k; q[k,c] += delta_k;
    q[k,~c] -= delta_k; b <- resp @ T(stay).
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_0 = ParameterSpec(
            "alpha_0", (0.0, 0.99), description="Learning rate, regime 0"
        )
        alpha_1 = ParameterSpec(
            "alpha_1", (0.0, 0.99), description="Learning rate, regime 1"
        )
        alpha_2 = ParameterSpec(
            "alpha_2", (0.0, 0.99), description="Learning rate, regime 2"
        )
        beta_0 = ParameterSpec(
            "beta_0", (0.1, 20.0), description="Inverse temp, regime 0"
        )
        beta_1 = ParameterSpec(
            "beta_1", (0.1, 20.0), description="Inverse temp, regime 1"
        )
        beta_2 = ParameterSpec(
            "beta_2", (0.1, 20.0), description="Inverse temp, regime 2"
        )
        stay = ParameterSpec(
            "stay",
            (0.0, 0.99),
            description="Probability of remaining in the same regime",
        )

    params: Params

    def reset(self):
        self.q = np.full((N_REGIMES, 2), 0.5)
        self.b = np.full(N_REGIMES, 1.0 / N_REGIMES)

    def forget(self):
        pass

    def get_state(self):
        return self.b.copy()

    def _regime_choice_probs(self):
        p = self.params
        betas = (p["beta_0"], p["beta_1"], p["beta_2"])
        return np.vstack([_softmax(self.q[k], betas[k]) for k in range(N_REGIMES)])

    def logits(self):
        opt_probs = self._regime_choice_probs()
        p_action = self.b @ opt_probs
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def update(self, choice, reward):
        opt_probs = self._regime_choice_probs()
        lik = opt_probs[:, choice]

        resp = self.b * lik
        resp_sum = resp.sum()
        if resp_sum <= 0:
            resp = np.full(N_REGIMES, 1.0 / N_REGIMES)
        else:
            resp /= resp_sum

        p = self.params
        alphas = (p["alpha_0"], p["alpha_1"], p["alpha_2"])
        other = 1 - choice
        for k in range(N_REGIMES):
            diff = self.q[k, choice] - self.q[k, other]
            rpe = reward - abs(diff)
            delta = alphas[k] * resp[k] * rpe
            self.q[k, choice] += delta
            self.q[k, other] -= delta
        np.clip(self.q, 0.0, 1.0, out=self.q)

        stay = p["stay"]
        switch = (1.0 - stay) / (N_REGIMES - 1)
        T = np.full((N_REGIMES, N_REGIMES), switch)
        np.fill_diagonal(T, stay)

        self.b = resp @ T
        self.b /= self.b.sum()


class QlearnDiff3StayRegime(BasePolicy):
    """
    'QlearnDiff1StayRegime' with per-regime persistence 'stay_0'/'stay_1'/
    'stay_2' instead of one shared 'stay'. A shared 'stay' forces uniform
    (1/3, 1/3, 1/3) occupancy regardless of its value; per-regime
    'stay_k' lifts that constraint. For this transition design
    (off-diagonal mass split uniformly), stationary occupancy has a
    closed form via flux balance:

        pi_k proportional to 1 / (1 - stay_k)     (mean dwell time)

    so a lopsided occupancy split (e.g. one dominant regime) is something
    the fit can discover without being told the task's true proportions.
    'occupancy()' returns this distribution.

    Per trial: same as 'QlearnDiff1StayRegime', but
    b <- resp @ T(stay_0, stay_1, stay_2).
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_0 = ParameterSpec(
            "alpha_0", (0.0, 0.99), description="Learning rate, regime 0"
        )
        alpha_1 = ParameterSpec(
            "alpha_1", (0.0, 0.99), description="Learning rate, regime 1"
        )
        alpha_2 = ParameterSpec(
            "alpha_2", (0.0, 0.99), description="Learning rate, regime 2"
        )
        beta_0 = ParameterSpec(
            "beta_0", (0.1, 20.0), description="Inverse temp, regime 0"
        )
        beta_1 = ParameterSpec(
            "beta_1", (0.1, 20.0), description="Inverse temp, regime 1"
        )
        beta_2 = ParameterSpec(
            "beta_2", (0.1, 20.0), description="Inverse temp, regime 2"
        )
        stay_0 = ParameterSpec(
            "stay_0", (0.0, 0.99), description="Probability of remaining in regime 0"
        )
        stay_1 = ParameterSpec(
            "stay_1", (0.0, 0.99), description="Probability of remaining in regime 1"
        )
        stay_2 = ParameterSpec(
            "stay_2", (0.0, 0.99), description="Probability of remaining in regime 2"
        )

    params: Params

    def reset(self):
        self.q = np.full((N_REGIMES, 2), 0.5)
        self.b = np.full(N_REGIMES, 1.0 / N_REGIMES)

    def forget(self):
        pass

    def get_state(self):
        return self.b.copy()

    def _regime_choice_probs(self):
        p = self.params
        betas = (p["beta_0"], p["beta_1"], p["beta_2"])
        return np.vstack([_softmax(self.q[k], betas[k]) for k in range(N_REGIMES)])

    def logits(self):
        opt_probs = self._regime_choice_probs()
        p_action = self.b @ opt_probs
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def _transition_matrix(self):
        p = self.params
        stays = (p["stay_0"], p["stay_1"], p["stay_2"])
        T = np.empty((N_REGIMES, N_REGIMES))
        for k in range(N_REGIMES):
            switch_k = (1.0 - stays[k]) / (N_REGIMES - 1)
            T[k, :] = switch_k
            T[k, k] = stays[k]
        return T

    def update(self, choice, reward):
        opt_probs = self._regime_choice_probs()
        lik = opt_probs[:, choice]

        resp = self.b * lik
        resp_sum = resp.sum()
        if resp_sum <= 0:
            resp = np.full(N_REGIMES, 1.0 / N_REGIMES)
        else:
            resp /= resp_sum

        p = self.params
        alphas = (p["alpha_0"], p["alpha_1"], p["alpha_2"])
        other = 1 - choice
        for k in range(N_REGIMES):
            diff = self.q[k, choice] - self.q[k, other]
            rpe = reward - abs(diff)
            delta = alphas[k] * resp[k] * rpe
            self.q[k, choice] += delta
            self.q[k, other] -= delta
        np.clip(self.q, 0.0, 1.0, out=self.q)

        self.b = resp @ self._transition_matrix()
        self.b /= self.b.sum()

    def occupancy(self):
        """Closed-form stationary occupancy, 'pi_k = 1/(1-stay_k)'
        normalized — long-run fraction of trials in each regime."""
        p = self.params
        inv_leave = np.array(
            [
                1.0 / (1.0 - p["stay_0"]),
                1.0 / (1.0 - p["stay_1"]),
                1.0 / (1.0 - p["stay_2"]),
            ]
        )
        return inv_leave / inv_leave.sum()


class MoARegime(BasePolicy):
    """
    Mixture-of-agents HMM (MoA-HMM), after Venditto, Miller, Brody & Daw
    ("Dynamic reinforcement learning reveals time-dependent shifts in
    strategy during reward learning"). Unlike the other classes here,
    regimes don't own private Q-values — a fixed set of agents (mbr, mbc,
    mfr, mfc, bias) updates *unconditionally* every trial regardless of
    current regime belief; only the weights combining them into a choice
    are regime-specific:

        V_k(y) = sum_A beta_A_k * Q_A(y);   p(y | regime k) = softmax(V_k)

    Agents:
    - mfr: Q *= (1-alpha_mfr); Q[c] += alpha_mfr * reward.
    - mfc: Q *= (1-alpha_mfc); Q[c] += alpha_mfc (ignores reward).
    - bias: fixed [+1, -1], never updated; weight 'beta_bias' is shared
      across regimes (a side preference is a stable trait, not something
      that should flip with strategy).
    - mbr/mbc: the paper's versions credit-assign via a "common vs. rare
      transition" flag from the two-step task, which has no analogue in a
      2-armed bandit (no intermediate state). Reinterpreted here with the
      coupled chosen/unchosen update from 'QlearnDiff', exploiting the
      structured task's known arm anti-correlation instead:

        pe_mbr = reward - Q_mbr[c]; Q_mbr[c] += alpha_mbr*pe_mbr; Q_mbr[~c] -= alpha_mbr*pe_mbr
        pe_mbc = 1 - Q_mbc[c];      Q_mbc[c] += alpha_mbc*pe_mbc; Q_mbc[~c] -= alpha_mbc*pe_mbc

      (mbc derived from mbr by swapping reward for a constant, the same
      way the paper derives mfc from mfr.) A deliberate reinterpretation,
      not a literal port.

    Transitions use a full 3x3 matrix — 'softmax([0, trans_k_1, trans_k_2])'
    per row k, regime 0 as reference — rather than a 'stay' parameter,
    matching the paper's asymmetric transition dynamics. 'occupancy()'
    solves for the stationary distribution numerically (no closed form for
    a general matrix).

    23 free params: 4 agent alphas (shared across regimes) + 4x3 regime
    combination weights + 1 shared beta_bias + 6 transition logits.
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_mbr = ParameterSpec(
            "alpha_mbr",
            (0.0, 0.99),
            description="Learning rate, model-based reward agent",
        )
        alpha_mbc = ParameterSpec(
            "alpha_mbc",
            (0.0, 0.99),
            description="Learning rate, model-based choice agent",
        )
        alpha_mfr = ParameterSpec(
            "alpha_mfr",
            (0.0, 0.99),
            description="Learning rate, model-free reward agent",
        )
        alpha_mfc = ParameterSpec(
            "alpha_mfc", (0.0, 0.99), description="Learning rate, choice-kernel agent"
        )
        beta_mbr_0 = ParameterSpec(
            "beta_mbr_0", (-10.0, 10.0), description="Regime 0 weight on mbr agent"
        )
        beta_mbc_0 = ParameterSpec(
            "beta_mbc_0", (-10.0, 10.0), description="Regime 0 weight on mbc agent"
        )
        beta_mfr_0 = ParameterSpec(
            "beta_mfr_0", (-10.0, 10.0), description="Regime 0 weight on mfr agent"
        )
        beta_mfc_0 = ParameterSpec(
            "beta_mfc_0", (-10.0, 10.0), description="Regime 0 weight on mfc agent"
        )
        beta_mbr_1 = ParameterSpec(
            "beta_mbr_1", (-10.0, 10.0), description="Regime 1 weight on mbr agent"
        )
        beta_mbc_1 = ParameterSpec(
            "beta_mbc_1", (-10.0, 10.0), description="Regime 1 weight on mbc agent"
        )
        beta_mfr_1 = ParameterSpec(
            "beta_mfr_1", (-10.0, 10.0), description="Regime 1 weight on mfr agent"
        )
        beta_mfc_1 = ParameterSpec(
            "beta_mfc_1", (-10.0, 10.0), description="Regime 1 weight on mfc agent"
        )
        beta_mbr_2 = ParameterSpec(
            "beta_mbr_2", (-10.0, 10.0), description="Regime 2 weight on mbr agent"
        )
        beta_mbc_2 = ParameterSpec(
            "beta_mbc_2", (-10.0, 10.0), description="Regime 2 weight on mbc agent"
        )
        beta_mfr_2 = ParameterSpec(
            "beta_mfr_2", (-10.0, 10.0), description="Regime 2 weight on mfr agent"
        )
        beta_mfc_2 = ParameterSpec(
            "beta_mfc_2", (-10.0, 10.0), description="Regime 2 weight on mfc agent"
        )
        beta_bias = ParameterSpec(
            "beta_bias",
            (-10.0, 10.0),
            description="Weight on bias agent (shared across regimes)",
        )
        trans_0_1 = ParameterSpec(
            "trans_0_1",
            (-10.0, 10.0),
            description="Logit, regime 0 -> regime 1 (ref: regime 0)",
        )
        trans_0_2 = ParameterSpec(
            "trans_0_2",
            (-10.0, 10.0),
            description="Logit, regime 0 -> regime 2 (ref: regime 0)",
        )
        trans_1_1 = ParameterSpec(
            "trans_1_1",
            (-10.0, 10.0),
            description="Logit, regime 1 -> regime 1 (ref: regime 0)",
        )
        trans_1_2 = ParameterSpec(
            "trans_1_2",
            (-10.0, 10.0),
            description="Logit, regime 1 -> regime 2 (ref: regime 0)",
        )
        trans_2_1 = ParameterSpec(
            "trans_2_1",
            (-10.0, 10.0),
            description="Logit, regime 2 -> regime 1 (ref: regime 0)",
        )
        trans_2_2 = ParameterSpec(
            "trans_2_2",
            (-10.0, 10.0),
            description="Logit, regime 2 -> regime 2 (ref: regime 0)",
        )

    params: Params

    def reset(self):
        self.q_mbr = np.full(2, 0.5)
        self.q_mbc = np.full(2, 0.5)
        self.q_mfr = np.zeros(2)
        self.q_mfc = np.zeros(2)
        self.q_bias = np.array([1.0, -1.0])
        self.b = np.full(N_REGIMES, 1.0 / N_REGIMES)

    def forget(self):
        pass

    def get_state(self):
        return self.b.copy()

    def _agent_values(self):
        return np.vstack([self.q_mbr, self.q_mbc, self.q_mfr, self.q_mfc])

    def _regime_betas(self):
        p = self.params
        return np.array(
            [
                [p["beta_mbr_0"], p["beta_mbc_0"], p["beta_mfr_0"], p["beta_mfc_0"]],
                [p["beta_mbr_1"], p["beta_mbc_1"], p["beta_mfr_1"], p["beta_mfc_1"]],
                [p["beta_mbr_2"], p["beta_mbc_2"], p["beta_mfr_2"], p["beta_mfc_2"]],
            ]
        )

    def _regime_choice_probs(self):
        betas = self._regime_betas()  # (N_REGIMES, N_AGENTS)
        agents = self._agent_values()  # (N_AGENTS, 2)
        # Broadcast-accumulate instead of `betas @ agents` (2D-by-2D `@`):
        # this environment's numpy/BLAS build hard-crashes (no traceback)
        # on small 2D matmuls, while 1D-by-2D products and elementwise ops
        # are unaffected — see `self.b @ opt_probs` below, which is safe.
        V = np.zeros((N_REGIMES, 2))
        for a in range(agents.shape[0]):
            V += betas[:, a : a + 1] * agents[a]
        V += self.params["beta_bias"] * self.q_bias  # shared across regimes
        return np.vstack([_softmax(V[k], 1.0) for k in range(N_REGIMES)])

    def logits(self):
        opt_probs = self._regime_choice_probs()
        p_action = self.b @ opt_probs
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def _transition_matrix(self):
        p = self.params
        # Row k = softmax([0, trans_k_1, trans_k_2]): regime 0 is a fixed
        # logit-0 reference destination, so this covers the full simplex
        # per row (a general, possibly asymmetric, transition matrix) with
        # only 2 free parameters per row.
        row_logits = np.array(
            [
                [0.0, p["trans_0_1"], p["trans_0_2"]],
                [0.0, p["trans_1_1"], p["trans_1_2"]],
                [0.0, p["trans_2_1"], p["trans_2_2"]],
            ]
        )
        return np.vstack([_softmax(row_logits[k], 1.0) for k in range(N_REGIMES)])

    def update(self, choice, reward):
        opt_probs = self._regime_choice_probs()
        lik = opt_probs[:, choice]

        resp = self.b * lik
        resp_sum = resp.sum()
        if resp_sum <= 0:
            resp = np.full(N_REGIMES, 1.0 / N_REGIMES)
        else:
            resp /= resp_sum

        p = self.params
        other = 1 - choice

        # unconditional agent updates — not scaled by resp
        pe_mbr = reward - self.q_mbr[choice]
        self.q_mbr[choice] += p["alpha_mbr"] * pe_mbr
        self.q_mbr[other] -= p["alpha_mbr"] * pe_mbr
        np.clip(self.q_mbr, 0.0, 1.0, out=self.q_mbr)

        pe_mbc = 1.0 - self.q_mbc[choice]
        self.q_mbc[choice] += p["alpha_mbc"] * pe_mbc
        self.q_mbc[other] -= p["alpha_mbc"] * pe_mbc
        np.clip(self.q_mbc, 0.0, 1.0, out=self.q_mbc)

        self.q_mfr *= 1.0 - p["alpha_mfr"]
        self.q_mfr[choice] += p["alpha_mfr"] * reward

        self.q_mfc *= 1.0 - p["alpha_mfc"]
        self.q_mfc[choice] += p["alpha_mfc"]

        self.b = resp @ self._transition_matrix()
        self.b /= self.b.sum()

    def occupancy(self):
        """Long-run regime occupancy — stationary distribution of the
        fitted transition matrix, solved numerically (no closed form for
        a general matrix)."""
        T = self._transition_matrix()
        eigvals, eigvecs = np.linalg.eig(T.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stat = np.real(eigvecs[:, idx])
        stat = np.clip(stat, 0.0, None)
        return stat / stat.sum()


class Qlearn2Regime(BasePolicy):
    """
    2-regime mixture-of-agents HMM — 'MoARegime's mechanism (fixed, shared
    agents that update unconditionally every trial; only the weights
    combining them into a choice are regime-specific), scaled down to 2
    agents and 2 regimes instead of 4 and 3.

    Agents: 'q1'/'q2' are two independent, symmetric 'Qlearn'-style
    submodels (own chosen/unchosen rates 'alpha_c_1'/'alpha_u_1' and
    'alpha_c_2'/'alpha_u_2'; standard counterfactual update, not the
    diff-based one) — unlike 'MoARegime's agents there's no built-in
    qualitative difference between them, that's left for the fit to
    discover via their regime-specific weights. 'bias' is a fixed
    [+1, -1] contrast, never updated, with a single weight 'beta_bias'
    shared across regimes (same mechanic as 'MoARegime's bias agent).

        V_k(y) = beta_q1_k*Q1(y) + beta_q2_k*Q2(y) + beta_bias*Q_bias(y)
        p(y | regime k) = softmax(V_k)

    With only 2 regimes, per-regime persistence 'stay_0'/'stay_1' is
    already the fully general transition matrix (nothing left to split
    when there's only one other regime to switch to), so the same
    closed-form occupancy as 'QlearnDiff3StayRegime' applies:
    'pi_k = 1/(1-stay_k)' normalized.

    Per trial: lik[k] = P(choice | regime k) from pre-update agent
    values; resp = normalize(b * lik); q1, q2 updated unconditionally
    (NOT scaled by resp — the key difference from 'Qlearn3Regime'-style
    classes) via pe = reward - q[c], q[c] += alpha_c*pe, q[~c] += alpha_u*pe;
    b <- resp @ [[stay_0, 1-stay_0], [1-stay_1, stay_1]].
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_c_1 = ParameterSpec(
            "alpha_c_1", (0.0, 0.99), description="Learning rate (chosen), agent 1"
        )
        alpha_u_1 = ParameterSpec(
            "alpha_u_1", (-0.99, 0.99), description="Learning rate (unchosen), agent 1"
        )
        alpha_c_2 = ParameterSpec(
            "alpha_c_2", (0.0, 0.99), description="Learning rate (chosen), agent 2"
        )
        alpha_u_2 = ParameterSpec(
            "alpha_u_2", (-0.99, 0.99), description="Learning rate (unchosen), agent 2"
        )
        beta_q1_0 = ParameterSpec(
            "beta_q1_0", (-10.0, 10.0), description="Regime 0 weight on agent 1"
        )
        beta_q2_0 = ParameterSpec(
            "beta_q2_0", (-10.0, 10.0), description="Regime 0 weight on agent 2"
        )
        beta_q1_1 = ParameterSpec(
            "beta_q1_1", (-10.0, 10.0), description="Regime 1 weight on agent 1"
        )
        beta_q2_1 = ParameterSpec(
            "beta_q2_1", (-10.0, 10.0), description="Regime 1 weight on agent 2"
        )
        beta_bias = ParameterSpec(
            "beta_bias",
            (-10.0, 10.0),
            description="Weight on bias agent (shared across regimes)",
        )
        stay_0 = ParameterSpec(
            "stay_0", (0.0, 0.99), description="Probability of remaining in regime 0"
        )
        stay_1 = ParameterSpec(
            "stay_1", (0.0, 0.99), description="Probability of remaining in regime 1"
        )

    params: Params

    def reset(self):
        self.q1 = np.full(2, 0.5)
        self.q2 = np.full(2, 0.5)
        self.q_bias = np.array([1.0, -1.0])
        self.b = np.full(2, 0.5)

    def forget(self):
        pass

    def get_state(self):
        return self.b.copy()

    def _agent_values(self):
        return np.vstack([self.q1, self.q2])

    def _regime_betas(self):
        p = self.params
        return np.array(
            [
                [p["beta_q1_0"], p["beta_q2_0"]],
                [p["beta_q1_1"], p["beta_q2_1"]],
            ]
        )

    def _regime_choice_probs(self):
        betas = self._regime_betas()  # (2, 2)
        agents = self._agent_values()  # (2, 2)
        # Broadcast-accumulate instead of `betas @ agents` — see
        # 'MoARegime._regime_choice_probs' for why (2D `@` crashes here).
        V = np.zeros((2, 2))
        for a in range(agents.shape[0]):
            V += betas[:, a : a + 1] * agents[a]
        V += self.params["beta_bias"] * self.q_bias
        return np.vstack([_softmax(V[k], 1.0) for k in range(2)])

    def logits(self):
        opt_probs = self._regime_choice_probs()
        p_action = self.b @ opt_probs
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def _transition_matrix(self):
        p = self.params
        s0, s1 = p["stay_0"], p["stay_1"]
        return np.array([[s0, 1.0 - s0], [1.0 - s1, s1]])

    def update(self, choice, reward):
        opt_probs = self._regime_choice_probs()
        lik = opt_probs[:, choice]

        resp = self.b * lik
        resp_sum = resp.sum()
        if resp_sum <= 0:
            resp = np.full(2, 0.5)
        else:
            resp /= resp_sum

        p = self.params
        other = 1 - choice

        # unconditional agent updates — not scaled by resp
        pe1 = reward - self.q1[choice]
        self.q1[choice] += p["alpha_c_1"] * pe1
        self.q1[other] += p["alpha_u_1"] * pe1
        np.clip(self.q1, 0.0, 1.0, out=self.q1)

        pe2 = reward - self.q2[choice]
        self.q2[choice] += p["alpha_c_2"] * pe2
        self.q2[other] += p["alpha_u_2"] * pe2
        np.clip(self.q2, 0.0, 1.0, out=self.q2)

        self.b = resp @ self._transition_matrix()
        self.b /= self.b.sum()

    def occupancy(self):
        """Closed-form stationary occupancy, 'pi_k = 1/(1-stay_k)'
        normalized — long-run fraction of trials in each regime."""
        p = self.params
        inv_leave = np.array([1.0 / (1.0 - p["stay_0"]), 1.0 / (1.0 - p["stay_1"])])
        return inv_leave / inv_leave.sum()


class RewardRateGatedRegime(BasePolicy):
    """
    2-regime switching model where regime membership is a DETERMINISTIC
    function of an observed covariate (a decaying reward-rate trace)
    rather than a latent state inferred via HMM belief propagation. No
    'b'-forward-filtering, no transition matrix, no 'stay' parameters —
    the gate is computed directly from data already observed, which
    breaks the label-switching symmetry that made 'Qlearn3Regime' collapse
    (a covariate-driven gate has no permutation-equivalent optima the way
    a free/blind transition matrix does).

    'reward_rate' is an exponential moving average of past rewards
    (updated causally — trial t's gate uses only rewards from before t):

        reward_rate(t+1) = rr_decay * reward_rate(t) + (1 - rr_decay) * reward(t)

    The gate is a sigmoid of reward-rate relative to a threshold:

        g(t) = sigmoid(gate_slope * (reward_rate(t) - gate_threshold))

    'g(t)' is the soft weight on the "high-reward-rate" regime ('q_high'),
    '1-g(t)' on the "low-reward-rate" regime ('q_low') — each a private,
    'Qlearn'-style Q-value pair (own 'alpha_c'/'alpha_u'/'beta'), updated
    every trial scaled by its gate weight (same resp-scaled mechanic as
    the frozen-Q regime classes, e.g. 'Qlearn3Regime' pre-realignment, but
    gated by the deterministic 'g' instead of an inferred posterior).

        p(y) = g(t)*softmax(q_high, beta_high)(y) + (1-g(t))*softmax(q_low, beta_low)(y)

    Per trial: g = sigmoid(gate_slope*(reward_rate - gate_threshold));
    pe_high = reward - q_high[c]; q_high[c] += alpha_c_high*g*pe_high;
    q_high[~c] += alpha_u_high*g*pe_high (symmetrically for q_low with
    1-g); reward_rate <- rr_decay*reward_rate + (1-rr_decay)*reward.
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_c_high = ParameterSpec(
            "alpha_c_high",
            (0.0, 0.99),
            description="Learning rate (chosen), high-reward-rate regime",
        )
        alpha_u_high = ParameterSpec(
            "alpha_u_high",
            (-0.99, 0.99),
            description="Learning rate (unchosen), high-reward-rate regime",
        )
        beta_high = ParameterSpec(
            "beta_high",
            (0.1, 20.0),
            description="Inverse temp, high-reward-rate regime",
        )
        alpha_c_low = ParameterSpec(
            "alpha_c_low",
            (0.0, 0.99),
            description="Learning rate (chosen), low-reward-rate regime",
        )
        alpha_u_low = ParameterSpec(
            "alpha_u_low",
            (-0.99, 0.99),
            description="Learning rate (unchosen), low-reward-rate regime",
        )
        beta_low = ParameterSpec(
            "beta_low", (0.1, 20.0), description="Inverse temp, low-reward-rate regime"
        )
        gate_slope = ParameterSpec(
            "gate_slope",
            (-50.0, 50.0),
            description="Sigmoid steepness of the reward-rate gate",
        )
        gate_threshold = ParameterSpec(
            "gate_threshold",
            (0.0, 1.0),
            description="Reward-rate value at which the gate is 0.5",
        )
        rr_decay = ParameterSpec(
            "rr_decay",
            (0.0, 0.99),
            description="Decay of the reward-rate trace (higher = longer memory)",
        )

    params: Params

    def reset(self):
        self.q_high = np.full(2, 0.5)
        self.q_low = np.full(2, 0.5)
        self.reward_rate = 0.5

    def forget(self):
        pass

    def _gate(self):
        p = self.params
        z = p["gate_slope"] * (self.reward_rate - p["gate_threshold"])
        return 1.0 / (1.0 + np.exp(-z))

    def get_state(self):
        g = self._gate()
        return np.array([g, 1.0 - g])

    def logits(self):
        p = self.params
        g = self._gate()
        p_high = _softmax(self.q_high, p["beta_high"])
        p_low = _softmax(self.q_low, p["beta_low"])
        p_action = g * p_high + (1.0 - g) * p_low
        p_action = np.clip(p_action, 1e-9, 1.0)
        return np.log(p_action)

    def update(self, choice, reward):
        p = self.params
        g = self._gate()
        other = 1 - choice

        pe_high = reward - self.q_high[choice]
        self.q_high[choice] += p["alpha_c_high"] * g * pe_high
        self.q_high[other] += p["alpha_u_high"] * g * pe_high
        np.clip(self.q_high, 0.0, 1.0, out=self.q_high)

        pe_low = reward - self.q_low[choice]
        self.q_low[choice] += p["alpha_c_low"] * (1.0 - g) * pe_low
        self.q_low[other] += p["alpha_u_low"] * (1.0 - g) * pe_low
        np.clip(self.q_low, 0.0, 1.0, out=self.q_low)

        self.reward_rate = (
            p["rr_decay"] * self.reward_rate + (1.0 - p["rr_decay"]) * reward
        )


class VolatilityGatedRegime(BasePolicy):
    """
    Single shared Q-value pair, single shared 'beta' — but the LEARNING
    RATE used to update Q is a soft blend of 3 regime-specific
    (alpha_c_k, alpha_u_k) pairs, gated by a running estimate of
    volatility rather than by belief inferred from choice likelihood.

    This is a structurally different family from 'Qlearn3Regime' and its
    relatives: there, regime identity is inferred from how well each
    regime's choice prediction matches the observed choice (Bayesian
    filtering, 'resp = normalize(b * lik)'). Here, 'logits()' depends
    only on the single shared 'q' and 'beta' — regime never affects the
    CURRENT choice prediction, only how strongly the CURRENT outcome
    updates 'q'. So there is no choice-likelihood signal to filter on;
    regime weight is instead driven entirely by how surprising recent
    OUTCOMES have been (a Pearce-Hall/Behrens-et-al.-style adaptive
    learning rate, discretized into 3 named alpha regimes instead of a
    single continuously-varying rate).

    Per trial, in order:
      1. 'logits()': choice comes from 'softmax(q, beta)' alone.
      2. After the choice/reward are observed, blend this trial's
         learning rate from the CURRENT gate weights 'w' (set at the end
         of the previous trial, i.e. reflecting volatility through
         trial t-1 — same causal convention as Pearce-Hall associability):
             alpha_c_eff = w @ [alpha_c_0, alpha_c_1, alpha_c_2]
             alpha_u_eff = w @ [alpha_u_0, alpha_u_1, alpha_u_2]
             pe = reward - q[choice]
             q[choice] += alpha_c_eff * pe; q[~choice] += alpha_u_eff * pe
      3. Update the running volatility estimate from this trial's own
         surprise (|pe|), one free parameter 'kappa':
             v <- v + kappa * (|pe| - v)
      4. Recompute 'w' from the new 'v' via a two-threshold soft gate
         (the same sigmoid mechanic as 'RewardRateGatedRegime', stacked
         twice to cover 3 categories instead of 2), ready for the next
         trial's blend:
             w_low  = 1 - sigmoid(slope * (v - thresh_1))
             w_high = sigmoid(slope * (v - thresh_2))
             w_mid  = 1 - w_low - w_high        (clipped >= 0, renormalized)

    'get_state()' returns 'w' — the regime-weight vector driving the
    NEXT update — which plays the same role 'b' played in the belief-
    based classes for downstream occupancy analysis (mean 'w' across
    trials, or fraction of trials where a regime is 'argmax(w)'). Unlike
    those classes, there is no closed-form stationary occupancy here
    ('v' isn't described by a fixed transition matrix), so occupancy
    must be estimated empirically from a fitted model's own trajectory
    rather than computed from the parameters directly — no 'occupancy()'
    method is provided.
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_c_0 = ParameterSpec(
            "alpha_c_0",
            (0.0, 0.99),
            description="Learning rate (chosen), regime 0 (low volatility)",
        )
        alpha_u_0 = ParameterSpec(
            "alpha_u_0",
            (-0.99, 0.99),
            description="Learning rate (unchosen), regime 0 (low volatility)",
        )
        alpha_c_1 = ParameterSpec(
            "alpha_c_1",
            (0.0, 0.99),
            description="Learning rate (chosen), regime 1 (mid volatility)",
        )
        alpha_u_1 = ParameterSpec(
            "alpha_u_1",
            (-0.99, 0.99),
            description="Learning rate (unchosen), regime 1 (mid volatility)",
        )
        alpha_c_2 = ParameterSpec(
            "alpha_c_2",
            (0.0, 0.99),
            description="Learning rate (chosen), regime 2 (high volatility)",
        )
        alpha_u_2 = ParameterSpec(
            "alpha_u_2",
            (-0.99, 0.99),
            description="Learning rate (unchosen), regime 2 (high volatility)",
        )
        beta = ParameterSpec(
            "beta", (0.1, 20.0), description="Inverse temperature (shared, single Q)"
        )
        kappa = ParameterSpec(
            "kappa",
            (0.0, 0.99),
            description="Volatility-tracking rate (EMA weight on new surprise)",
        )
        thresh_1 = ParameterSpec(
            "thresh_1", (0.0, 1.0), description="Volatility threshold: low/mid boundary"
        )
        thresh_2 = ParameterSpec(
            "thresh_2",
            (0.0, 1.0),
            description="Volatility threshold: mid/high boundary",
        )
        slope = ParameterSpec(
            "slope", (0.0, 50.0), description="Steepness of the volatility gate"
        )

    params: Params

    def reset(self):
        self.q = np.full(2, 0.5)
        self.v = 0.0
        self.w = self._gate(self.v)

    def forget(self):
        pass

    def get_state(self):
        return self.w.copy()

    def _gate(self, v):
        p = self.params
        w_low = 1.0 - _sigmoid(p["slope"] * (v - p["thresh_1"]))
        w_high = _sigmoid(p["slope"] * (v - p["thresh_2"]))
        w_mid = 1.0 - w_low - w_high
        w = np.clip(np.array([w_low, w_mid, w_high]), 0.0, None)
        total = w.sum()
        return w / total if total > 0 else np.full(3, 1.0 / 3.0)

    def logits(self):
        p = self.params
        probs = _softmax(self.q, p["beta"])
        return np.log(np.clip(probs, 1e-9, 1.0))

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice

        alphas_c = np.array([p["alpha_c_0"], p["alpha_c_1"], p["alpha_c_2"]])
        alphas_u = np.array([p["alpha_u_0"], p["alpha_u_1"], p["alpha_u_2"]])
        alpha_c_eff = self.w @ alphas_c
        alpha_u_eff = self.w @ alphas_u

        pe = reward - self.q[choice]
        self.q[choice] += alpha_c_eff * pe
        self.q[other] += alpha_u_eff * pe
        np.clip(self.q, 0.0, 1.0, out=self.q)

        surprise = abs(pe)
        self.v = self.v + p["kappa"] * (surprise - self.v)
        self.w = self._gate(self.v)


class SupervisedMoERegime(BasePolicy):
    """
    Mixture-of-experts gated on a KNOWN, OBSERVED block-type covariate 'D'
    (e.g. low_low/high_low/high_high tier) supplied by the caller, NOT a
    latent state inferred from choice/reward. Three independent Qlearn-
    style experts ('q0'/'q1'/'q2', own 'alpha_c_i'/'alpha_u_i', shared
    'beta') update unconditionally every trial (same always-learning
    mechanic as 'Qlearn2Regime'), combined via softmax gate weights that
    depend only on 'D':

        w_i(D) = softmax(gate_logit[D, :])_i
        p(y) = sum_i w_i(D) * softmax(q_i, beta)(y)

    Because 'D' is observed rather than inferred, there is NO belief
    propagation, no HMM, no forward-filtering, and none of the
    identifiability/collapse failure modes documented for 'Qlearn3Regime'
    and its relatives (see that class's docstring/history) — fitting this
    is a plain supervised MLE problem, as well-behaved as fitting any
    ordinary multinomial-logistic-gated regression. It answers a
    narrower, different question than the latent models: "does behavior
    differ by KNOWN block type," not "does the animal's own inferred
    state track block type."

    'D' must be precomputed by the caller as one integer per trial in
    {0, 1, 2} (matching the task's true trial order) and passed via
    `d_sequence` at construction. It is consumed through an internal
    cursor 'self._pos', advanced once per trial in `update()`. That
    cursor is reset to 0 in `set_params()` — which 'DecisionModel' calls
    exactly once per evaluated parameter vector — rather than in
    `reset()`, which ALSO fires at every session/window boundary
    mid-sequence and must leave the cursor alone so it keeps indexing the
    trial's absolute position in 'd_sequence', not a per-session position.
    """

    default_beta_schedule = NoBeta

    def __init__(self, d_sequence, **kwargs):
        self.d_sequence = np.asarray(d_sequence, dtype=int)
        super().__init__(**kwargs)

    class Params(ParameterGroup):
        alpha_c_0 = ParameterSpec(
            "alpha_c_0", (0.0, 0.99), description="Learning rate (chosen), expert 0"
        )
        alpha_u_0 = ParameterSpec(
            "alpha_u_0", (-0.99, 0.99), description="Learning rate (unchosen), expert 0"
        )
        alpha_c_1 = ParameterSpec(
            "alpha_c_1", (0.0, 0.99), description="Learning rate (chosen), expert 1"
        )
        alpha_u_1 = ParameterSpec(
            "alpha_u_1", (-0.99, 0.99), description="Learning rate (unchosen), expert 1"
        )
        alpha_c_2 = ParameterSpec(
            "alpha_c_2", (0.0, 0.99), description="Learning rate (chosen), expert 2"
        )
        alpha_u_2 = ParameterSpec(
            "alpha_u_2", (-0.99, 0.99), description="Learning rate (unchosen), expert 2"
        )
        beta = ParameterSpec(
            "beta", (0.1, 20.0), description="Inverse temperature (shared across experts)"
        )
        gate_0_1 = ParameterSpec(
            "gate_0_1", (-10.0, 10.0), description="D=0 gate logit, expert 1 (ref: expert 0)"
        )
        gate_0_2 = ParameterSpec(
            "gate_0_2", (-10.0, 10.0), description="D=0 gate logit, expert 2 (ref: expert 0)"
        )
        gate_1_1 = ParameterSpec(
            "gate_1_1", (-10.0, 10.0), description="D=1 gate logit, expert 1 (ref: expert 0)"
        )
        gate_1_2 = ParameterSpec(
            "gate_1_2", (-10.0, 10.0), description="D=1 gate logit, expert 2 (ref: expert 0)"
        )
        gate_2_1 = ParameterSpec(
            "gate_2_1", (-10.0, 10.0), description="D=2 gate logit, expert 1 (ref: expert 0)"
        )
        gate_2_2 = ParameterSpec(
            "gate_2_2", (-10.0, 10.0), description="D=2 gate logit, expert 2 (ref: expert 0)"
        )

    params: Params

    def set_params(self, params):
        super().set_params(params)
        self._pos = 0

    def reset(self):
        self.q0 = np.full(2, 0.5)
        self.q1 = np.full(2, 0.5)
        self.q2 = np.full(2, 0.5)
        # self._pos intentionally NOT touched here — see class docstring.

    def forget(self):
        pass

    def _agent_values(self):
        return np.vstack([self.q0, self.q1, self.q2])

    def _current_weights(self):
        d = int(self.d_sequence[self._pos])
        p = self.params
        gate_rows = {
            0: (0.0, p["gate_0_1"], p["gate_0_2"]),
            1: (0.0, p["gate_1_1"], p["gate_1_2"]),
            2: (0.0, p["gate_2_1"], p["gate_2_2"]),
        }
        return _softmax(np.array(gate_rows[d]), 1.0)

    def get_state(self):
        return self._current_weights()

    def logits(self):
        w = self._current_weights()
        agents = self._agent_values()
        beta = self.params["beta"]
        opt_probs = np.vstack([_softmax(agents[i], beta) for i in range(3)])
        p_action = w @ opt_probs
        return np.log(np.clip(p_action, 1e-9, 1.0))

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice
        for q, ac, au in (
            (self.q0, "alpha_c_0", "alpha_u_0"),
            (self.q1, "alpha_c_1", "alpha_u_1"),
            (self.q2, "alpha_c_2", "alpha_u_2"),
        ):
            pe = reward - q[choice]
            q[choice] += p[ac] * pe
            q[other] += p[au] * pe
            np.clip(q, 0.0, 1.0, out=q)
        self._pos += 1
