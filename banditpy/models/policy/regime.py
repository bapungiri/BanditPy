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


class QlearnRegime(BasePolicy):
    """
    3-regime switching Q-learning: an unsupervised HMM over latent
    behavioral regimes, each with its own Q-learning sub-policy.

    No block-type or condition label is ever given to the model — which
    regime is active on a given trial is inferred purely from the choice/
    reward sequence. Each regime 'k' holds its own action values 'q[k]' and
    owns its own chosen/unchosen learning rates 'alpha_c_k'/'alpha_u_k'
    (same counterfactual update rule as 'Qlearn') and inverse temperature
    'beta_k'; a belief 'b' over the 3 regimes is propagated trial-to-trial
    through a symmetric Markov transition matrix governed by a single
    persistence parameter 'stay' (probability of remaining in the same
    regime). Matching 'Qlearn's per-regime update rule (rather than a
    simpler one) means this model is a proper generalization of it: with
    'stay' pinned near 1 (no switching), a single regime reduces exactly to
    'Qlearn', so a fitted improvement from 3 regimes over 1 can be
    attributed to switching itself rather than to a simultaneously
    different learning rule.

    'logits()' returns the log of the belief-weighted mixture of each
    regime's softmax choice probabilities, so this is 'NoBeta' by default
    (see 'QlearnHierarchical' for the same pattern). Because the belief
    update in 'update()' is the standard HMM forward-filtering recursion,
    the sum of per-trial log 'logits()' probabilities computed by
    'DecisionModel' is exactly the HMM's marginal log-likelihood — no
    separate EM loop is needed, and this fits directly into the existing
    'fit()' / 'cross_validate()' machinery.

    Note
    ----
    'stay' is shared across all 3 regimes, which forces the model's
    long-run occupancy of each regime to be exactly uniform (1/3 each)
    regardless of 'stay's fitted value — see 'QlearnRegimeDiffStays' for a
    per-regime persistence parameterization that lifts this constraint.

    'update()' order per trial:
    1. lik[k]     = P(choice | regime k's current Q, beta_k)
    2. resp       = normalize(b * lik)                         # posterior over regimes
    3. q[k, c]   += alpha_c_k * resp[k] * (reward - q[k, c])    # chosen, per-regime RL update
    4. q[k, ~c]  += alpha_u_k * resp[k] * (reward - q[k, c])    # unchosen, per-regime RL update
    5. b         <- resp @ T(stay)                              # propagate belief
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_c_0 = ParameterSpec(
            "alpha_c_0", (0.0, 0.99), description="Learning rate (chosen), regime 0"
        )
        alpha_u_0 = ParameterSpec(
            "alpha_u_0", (-0.99, 0.99), description="Learning rate (unchosen), regime 0"
        )
        alpha_c_1 = ParameterSpec(
            "alpha_c_1", (0.0, 0.99), description="Learning rate (chosen), regime 1"
        )
        alpha_u_1 = ParameterSpec(
            "alpha_u_1", (-0.99, 0.99), description="Learning rate (unchosen), regime 1"
        )
        alpha_c_2 = ParameterSpec(
            "alpha_c_2", (0.0, 0.99), description="Learning rate (chosen), regime 2"
        )
        alpha_u_2 = ParameterSpec(
            "alpha_u_2", (-0.99, 0.99), description="Learning rate (unchosen), regime 2"
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
        alphas_c = (p["alpha_c_0"], p["alpha_c_1"], p["alpha_c_2"])
        alphas_u = (p["alpha_u_0"], p["alpha_u_1"], p["alpha_u_2"])
        other = 1 - choice
        for k in range(N_REGIMES):
            pe = reward - self.q[k, choice]
            self.q[k, choice] += alphas_c[k] * resp[k] * pe
            self.q[k, other] += alphas_u[k] * resp[k] * pe
        np.clip(self.q, 0.0, 1.0, out=self.q)

        stay = p["stay"]
        switch = (1.0 - stay) / (N_REGIMES - 1)
        T = np.full((N_REGIMES, N_REGIMES), switch)
        np.fill_diagonal(T, stay)

        self.b = resp @ T
        self.b /= self.b.sum()


class QlearnRegimeDiff(BasePolicy):
    """
    3-regime switching Q-learning driven by the chosen-vs-unchosen value
    difference (see 'QlearnDiff'), rather than separate chosen/unchosen
    learning rates.

    Same latent-regime / belief-propagation machinery as 'QlearnRegime',
    but each regime owns a single learning rate 'alpha_k' instead of an
    'alpha_c_k'/'alpha_u_k' pair — 'QlearnDiff's difference-based update
    still moves both options (in opposite directions) on every trial, so
    dropping the second per-regime parameter does not remove the
    counterfactual update, it just ties the chosen/unchosen movement to a
    common rate (7 free parameters total vs. 10 for 'QlearnRegime'). With
    'stay' pinned near 1 (no switching), a single regime reduces exactly
    to 'QlearnDiff'.

    Note
    ----
    'stay' is shared across all 3 regimes, which forces the model's
    long-run occupancy of each regime to be exactly uniform (1/3 each)
    regardless of 'stay's fitted value — see 'QlearnRegimeDiffStays' for a
    per-regime persistence parameterization that lifts this constraint.

    'update()' order per trial:
    1. lik[k]    = P(choice | regime k's current Q, beta_k)
    2. resp      = normalize(b * lik)                    # posterior over regimes
    3. diff_k    = q[k, c] - q[k, ~c]
       rpe_k     = reward - abs(diff_k)
       delta_k   = alpha_k * resp[k] * rpe_k
       q[k, c]  += delta_k
       q[k, ~c] -= delta_k
    4. b         <- resp @ T(stay)                        # propagate belief
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


class QlearnRegimeDiffStays(BasePolicy):
    """
    'QlearnRegimeDiff' with a per-regime persistence parameter
    'stay_0'/'stay_1'/'stay_2' instead of one shared 'stay'.

    A shared 'stay' forces every regime to have identical dwell-time
    statistics, which in turn forces the model's long-run occupancy of
    each regime to be exactly uniform (1/3, 1/3, 1/3) no matter what value
    'stay' takes. Giving each regime its own persistence lifts that
    constraint: for this transition-matrix design (off-diagonal mass split
    uniformly across the other regimes), a flux-balance argument gives a
    closed form for the stationary occupancy,

        pi_k  proportional to  1 / (1 - stay_k)

    i.e. proportional to each regime's mean dwell time. A regime with
    stay_k around 0.9 will dominate occupancy over ones with stay_k around
    0.5, so a lopsided occupancy split (e.g. one dominant regime, matching
    a task where one block-type combination is far more frequent than the
    others) is something this parameterization can express and the fit can
    discover — without ever being told the task's block-type proportions.
    After fitting, 'occupancy()' returns this closed-form distribution,
    to be compared against the empirical time-averaged belief and against
    the task's true condition frequencies.

    'update()' order per trial:
    1. lik[k]    = P(choice | regime k's current Q, beta_k)
    2. resp      = normalize(b * lik)                    # posterior over regimes
    3. diff_k    = q[k, c] - q[k, ~c]
       rpe_k     = reward - abs(diff_k)
       delta_k   = alpha_k * resp[k] * rpe_k
       q[k, c]  += delta_k
       q[k, ~c] -= delta_k
    4. b         <- resp @ T(stay_0, stay_1, stay_2)      # propagate belief
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
        """Closed-form stationary regime occupancy, 'pi_k = 1/(1-stay_k)'
        normalized to sum to 1 — the model's implied long-run fraction of
        trials spent in each regime, independent of any particular
        session's trial count or starting belief.
        """
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
    (2024, "Dynamic reinforcement learning reveals time-dependent shifts in
    strategy during reward learning") — architecturally distinct from the
    other classes in this file.

    'QlearnRegime'/'QlearnRegimeDiff'/'QlearnRegimeDiffStays' give each
    regime its own private Q-values, updated only in proportion to that
    regime's current responsibility ('resp[k]') — so a regime's values sit
    frozen whenever it's not currently believed active. This class instead
    follows the paper's structure: a small FIXED set of agents, each with
    its own persistent values, updates *unconditionally* every trial
    regardless of which regime is currently believed active — only the
    weights used to *combine* the agents' values into a choice change with
    regime. Concretely, per regime k:

        V_k(y) = sum_A  beta_A_k * Q_A(y)
        p(y | regime k) = softmax(V_k)

    and the belief 'b' over regimes marginalizes over this per trial,
    exactly as in the other regime classes.

    Agents (first draft — meant to be tinkered with)
    --------------------------------------------------
    - 'mfr' (model-free, reward-driven): Q *= (1-alpha_mfr); Q[c] += alpha_mfr * reward.
    - 'mfc' (model-free, choice-driven / perseveration): Q *= (1-alpha_mfc); Q[c] += alpha_mfc.
      Ignores reward entirely — a pure choice kernel, after the paper's 'MFchoice' agent.
    - 'bias' (static side preference): fixed contrast [+1, -1], never
      updated — matches the paper's 'Bias' agent (no learning rate). Unlike
      'mfr'/'mfc', its weight 'beta_bias' is a single value shared across
      all 3 regimes rather than regime-specific — a side preference is a
      stable trait, not obviously something that should flip with inferred
      strategy, and every other class in this file already treats bias as
      global rather than regime-dependent.

    Only 2 agents for now — the paper's model-based agents rely on the
    two-step task's action-state transition model, which has no analogue
    in a 2-armed bandit, so that piece is left out here rather than
    guessed at. A bandit-appropriate stand-in (e.g. an agent that assumes
    the two arms' probabilities are anti-correlated and updates the
    unchosen arm oppositely, exploiting the structured task's known
    correlation) can be added back in once you've settled on what
    "model-based" should mean for this task — see 'QlearnRegimeDiff's
    coupled chosen/unchosen update for the update rule it would reuse.

    Agent learning rates ('alpha_mfr'/'alpha_mfc') are shared across
    regimes, matching the paper (only the combination weights are
    regime-specific) — 2 + 3*2 + 1 + 1 = 10 free parameters total.

    'update()' order per trial:
    1. V_k, opt_probs[k]  = combined values / choice probs per regime (pre-update agent values)
    2. lik[k]             = opt_probs[k, choice]
    3. resp               = normalize(b * lik)                    # posterior over regimes (belief only)
    4. q_mfr, q_mfc       += unconditional per-agent updates       # NOT scaled by resp — the key
                                                                    # difference from the other classes
    5. b                  <- resp @ T(stay)                       # propagate belief
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_mfr = ParameterSpec(
            "alpha_mfr",
            (0.0, 0.99),
            description="Learning rate, model-free reward agent",
        )
        alpha_mfc = ParameterSpec(
            "alpha_mfc", (0.0, 0.99), description="Learning rate, choice-kernel agent"
        )
        beta_mfr_0 = ParameterSpec(
            "beta_mfr_0", (-10.0, 10.0), description="Regime 0 weight on mfr agent"
        )
        beta_mfc_0 = ParameterSpec(
            "beta_mfc_0", (-10.0, 10.0), description="Regime 0 weight on mfc agent"
        )
        beta_mfr_1 = ParameterSpec(
            "beta_mfr_1", (-10.0, 10.0), description="Regime 1 weight on mfr agent"
        )
        beta_mfc_1 = ParameterSpec(
            "beta_mfc_1", (-10.0, 10.0), description="Regime 1 weight on mfc agent"
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
        stay = ParameterSpec(
            "stay",
            (0.0, 0.99),
            description="Probability of remaining in the same regime",
        )

    params: Params

    def reset(self):
        self.q_mfr = np.zeros(2)
        self.q_mfc = np.zeros(2)
        self.q_bias = np.array([1.0, -1.0])
        self.b = np.full(N_REGIMES, 1.0 / N_REGIMES)

    def forget(self):
        pass

    def _agent_values(self):
        return np.vstack([self.q_mfr, self.q_mfc])

    def _regime_betas(self):
        p = self.params
        return np.array(
            [
                [p["beta_mfr_0"], p["beta_mfc_0"]],
                [p["beta_mfr_1"], p["beta_mfc_1"]],
                [p["beta_mfr_2"], p["beta_mfc_2"]],
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

        # unconditional agent updates — not scaled by resp
        self.q_mfr *= 1.0 - p["alpha_mfr"]
        self.q_mfr[choice] += p["alpha_mfr"] * reward

        self.q_mfc *= 1.0 - p["alpha_mfc"]
        self.q_mfc[choice] += p["alpha_mfc"]

        stay = p["stay"]
        switch = (1.0 - stay) / (N_REGIMES - 1)
        T = np.full((N_REGIMES, N_REGIMES), switch)
        np.fill_diagonal(T, stay)

        self.b = resp @ T
        self.b /= self.b.sum()
