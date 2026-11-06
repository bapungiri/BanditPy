import math
import numpy as np
from banditpy.models.policy.base import BasePolicy, ParameterGroup, ParameterSpec
from .beta_schedule import NoBeta


def _softmax(x: np.ndarray, beta: float) -> np.ndarray:
    z = beta * x
    z -= z.max()
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        return np.full_like(x, 1.0 / len(x))
    return e / s


class Qlearn(BasePolicy):
    """
    2-arm Q-learning with counterfactual updates, a port bias term, an
    optional perseverance (sticky) term and an optional decay of the
    unchosen value.

    Update rule:
    Q[choice] += alpha_c * (reward - Q[choice])
    Q[unchosen] += alpha_u * (reward - Q[choice])
    h += alpha_h * (choice - h)          # choice trace, 0 = arm 0, 1 = arm 1

    At each reset Q = [q_init, q_init] and h = 0.5. Between trials (not at a
    reset), the value of the arm not chosen on the previous trial decays
    toward q_init:
    Q[unchosen] += decay * (q_init - Q[unchosen])

    Choice values, softmax with the beta schedule's 'beta':
    V[0] = beta * (Q[0] + bias) + sticky * (0.5 - h)
    V[1] = beta * (Q[1] - bias) + sticky * (h - 0.5)

    'sticky' sits outside beta, so it is the perseverance strength itself
    rather than a multiple of beta (which left beta and sticky trading off
    in fits). It is >= 0: it favours repeating recent choices (perseverance
    only; no alternation).
    All six parameters are fitted by default: animals repeat their last
    choice far more than value learning alone predicts. Disable
    parameters to fit other variants, e.g.::

        policy = Qlearn()
        policy.params.alpha_h.disable()
        policy.params.sticky.disable()  # plain Q-learning with a bias
        policy.params.bias.disable()    # without port bias (bias = 0)

    A disabled 'sticky' is 0, which makes the choice trace a no-op.
    'decay' is off by default (0, no decay); enable it with
    'policy.params.decay.enable()'. It acts on the Q-values only, not on the
    choice trace. 'q_init' is fixed at 0.5 unless enabled; both ports start
    at it, so without decay it only shifts the first prediction errors of
    each reset and is weakly identifiable. Enable it together with decay.
    """

    class Params(ParameterGroup):
        alpha_c = ParameterSpec(
            "alpha_c", (0.0, 0.99), description="Learning rate for chosen option"
        )
        alpha_u = ParameterSpec(
            "alpha_u", (-0.99, 0.99), description="Learning rate for unchosen option"
        )
        bias = ParameterSpec(
            "bias", (-1.0, 1.0), default=0.0, description="Bias toward port 0 vs port 1"
        )
        alpha_h = ParameterSpec(
            "alpha_h",
            (0.0, 1.0),
            default=0.0,
            active=True,
            description="Learning rate of the choice (perseverance) trace",
        )
        sticky = ParameterSpec(
            "sticky",
            (0.0, 20.0),
            default=0.0,
            active=True,
            description="Perseverance weight, outside beta (repeat recent choices)",
        )
        decay = ParameterSpec(
            "decay",
            (0.0, 1.0),
            default=0.0,
            active=False,
            description="Decay of the unchosen Q-value toward q_init per trial",
        )
        q_init = ParameterSpec(
            "q_init",
            (0.0, 1.0),
            default=0.5,
            active=False,
            description="Q-values at each reset, and the decay target",
        )

    params: Params

    # Plain Python floats instead of 2-element numpy arrays: these run once
    # per trial in every NLL evaluation, where numpy's per-call overhead
    # (np.clip, array construction) dominates the arithmetic.

    def reset(self):
        q0 = self.params["q_init"]
        self.q = [q0, q0]
        self.h = 0.5
        self.last_choice = None

    def forget(self):
        d = self.params["decay"]
        if d and self.last_choice is not None:
            other = 1 - self.last_choice
            self.q[other] += d * (self.params["q_init"] - self.q[other])

    def logits(self):
        # The caller scales the logits by beta; dividing the sticky term by
        # beta here leaves it unscaled in the softmax.
        b = self.params["bias"]
        stick = self.params["sticky"] * (self.h - 0.5) / self.beta_schedule.get_beta()
        return (self.q[0] + b - stick, self.q[1] - b + stick)

    def update(self, choice, reward):
        a_c = self.params["alpha_c"]
        a_u = self.params["alpha_u"]

        other = 1 - choice
        pe = reward - self.q[choice]

        q_c = self.q[choice] + a_c * pe
        q_o = self.q[other] + a_u * pe
        self.q[choice] = min(max(q_c, 0.0), 1.0)
        self.q[other] = min(max(q_o, 0.0), 1.0)

        self.h += self.params["alpha_h"] * (choice - self.h)
        self.last_choice = choice


class QlearnCost(BasePolicy):
    """
    Average-reward ('relative value') Q-learning: outcomes are judged against a
    learned average reward rate 'rho', the opportunity cost of a trial
    (R-learning, Schwartz 1993; Daw & Touretzky 2002).

    Reward rate, updated on every trial:

        rho <- rho + alpha_rho * (reward - rho)

    Prediction error relative to rho, and 'Qlearn' value updates:

        delta        = (reward - rho) - Q[choice]
        Q[choice]   += alpha_c * delta
        Q[unchosen] += alpha_u * delta          (both clipped to [-rho, 1 - rho])
        h           += alpha_h * (choice - h)

    Choice, as in 'Qlearn' (sticky outside beta):

        V[0] = beta * (Q[0] + bias) + sticky * (0.5 - h)
        V[1] = beta * (Q[1] - bias) + sticky * (h - 0.5)

    Q now means better or worse than the environment's average, so values can
    be negative: they are clipped to the range of reward - rho, [-rho, 1 - rho]
    (an earlier [-1, 1] clip did not nest 'Qlearn': with alpha_u < 0 the
    unchosen value could go below 0). At each reset Q = [0.5, 0.5], h = 0.5
    and rho = 'rho0' (fixed at 0 unless enabled), so with alpha_rho = 0 this
    is exactly 'Qlearn' (QlearnSticky): the extra parameter alpha_rho tests
    whether a learned reward rate helps. Because rho shifts both ports' values by the same amount,
    it changes choices only through how the value levels feed the next update
    and through clipping: a weak handle, so check recovery before reading it.
    Default active parameters: alpha_c, alpha_u, bias, alpha_h, sticky,
    alpha_rho (+ beta from the beta schedule); 'rho0' is off.
    """

    class Params(ParameterGroup):
        alpha_c = ParameterSpec(
            "alpha_c", (0.0, 0.99), description="Learning rate for chosen option"
        )
        alpha_u = ParameterSpec(
            "alpha_u", (-0.99, 0.99), description="Learning rate for unchosen option"
        )
        bias = ParameterSpec(
            "bias", (-1.0, 1.0), default=0.0, description="Bias toward port 0 vs port 1"
        )
        alpha_h = ParameterSpec(
            "alpha_h",
            (0.0, 1.0),
            default=0.0,
            description="Learning rate of the choice (perseverance) trace",
        )
        sticky = ParameterSpec(
            "sticky",
            (0.0, 20.0),
            default=0.0,
            description="Perseverance weight, outside beta (repeat recent choices)",
        )
        alpha_rho = ParameterSpec(
            "alpha_rho",
            (0.0, 1.0),
            default=0.0,
            description="Learning rate of the average reward rate rho",
        )
        rho0 = ParameterSpec(
            "rho0",
            (0.0, 1.0),
            default=0.0,
            active=False,
            description="Average reward rate rho at each reset (0 = nests Qlearn)",
        )

    params: Params

    # Plain Python floats, as in 'Qlearn' (per-trial code in every NLL evaluation).

    def reset(self):
        self.q = [0.5, 0.5]
        self.h = 0.5
        self.rho = self.params["rho0"]

    def forget(self):
        pass

    def logits(self):
        # Sticky outside beta, as in 'Qlearn'.
        b = self.params["bias"]
        stick = self.params["sticky"] * (self.h - 0.5) / self.beta_schedule.get_beta()
        return (self.q[0] + b - stick, self.q[1] - b + stick)

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice
        delta = (reward - self.rho) - self.q[choice]
        # Clip to the range of reward - rho (reward in [0, 1]); at rho = 0 this
        # is Qlearn's [0, 1].
        lo, hi = -self.rho, 1.0 - self.rho
        self.q[choice] = min(max(self.q[choice] + p["alpha_c"] * delta, lo), hi)
        self.q[other] = min(max(self.q[other] + p["alpha_u"] * delta, lo), hi)
        self.h += p["alpha_h"] * (choice - self.h)
        self.rho += p["alpha_rho"] * (reward - self.rho)


class QlearnMVTHybrid(BasePolicy):
    """
    'Qlearn' in which the port not chosen on the last trial is valued partly
    by the environment's reward rate 'rho' (marginal value theorem: the value
    of leaving is what the environment offers on average; Charnov 1976,
    Constantino & Daw 2015).

    Values and choice trace are updated as in 'Qlearn' (Q clipped to [0, 1]),
    and the reward rate on every trial:

        rho <- rho + alpha_rho * (reward - rho)

    Choice, with c the port chosen on the last trial and o the other one:

        W[c] = Q[c]
        W[o] = w * Q[o] + (1 - w) * rho
        V[0] = beta * (W[0] + bias) + sticky * (0.5 - h)
        V[1] = beta * (W[1] - bias) + sticky * (h - 0.5)

    A rich environment raises the value of leaving, so the animal switches
    more after a run of rewards on either port. With w = 1 this is exactly
    'Qlearn' (QlearnSticky). On the first trial after a reset, W = Q.
    At each reset Q = [0.5, 0.5], h = 0.5 and rho = 'rho0' (fixed at 0.5
    unless enabled). With alpha_rho = 0, rho stays at rho0 and w < 1 only
    shrinks the other port's value toward 0.5, so compare fits with
    alpha_rho free and fixed at 0 to test the reward rate itself.
    Default active parameters: alpha_c, alpha_u, bias, alpha_h, sticky, w,
    alpha_rho (+ beta from the beta schedule); 'rho0' is off.
    """

    class Params(ParameterGroup):
        alpha_c = ParameterSpec(
            "alpha_c", (0.0, 0.99), description="Learning rate for chosen option"
        )
        alpha_u = ParameterSpec(
            "alpha_u", (-0.99, 0.99), description="Learning rate for unchosen option"
        )
        bias = ParameterSpec(
            "bias", (-1.0, 1.0), default=0.0, description="Bias toward port 0 vs port 1"
        )
        alpha_h = ParameterSpec(
            "alpha_h",
            (0.0, 1.0),
            default=0.0,
            description="Learning rate of the choice (perseverance) trace",
        )
        sticky = ParameterSpec(
            "sticky",
            (0.0, 20.0),
            default=0.0,
            description="Perseverance weight, outside beta (repeat recent choices)",
        )
        w = ParameterSpec(
            "w",
            (0.0, 1.0),
            default=1.0,
            description="Weight of the other port's Q vs rho in its value (1 = Qlearn)",
        )
        alpha_rho = ParameterSpec(
            "alpha_rho",
            (0.0, 1.0),
            default=0.0,
            description="Learning rate of the environment reward rate rho",
        )
        rho0 = ParameterSpec(
            "rho0",
            (0.0, 1.0),
            default=0.5,
            active=False,
            description="Environment reward rate rho at each reset",
        )

    params: Params

    # Plain Python floats, as in 'Qlearn' (per-trial code in every NLL evaluation).

    def reset(self):
        self.q = [0.5, 0.5]
        self.h = 0.5
        self.rho = self.params["rho0"]
        self.last_choice = None

    def forget(self):
        pass

    def logits(self):
        # Sticky outside beta, as in 'Qlearn'.
        p = self.params
        v = list(self.q)
        if self.last_choice is not None:
            o = 1 - self.last_choice
            v[o] = p["w"] * v[o] + (1.0 - p["w"]) * self.rho
        b = p["bias"]
        stick = p["sticky"] * (self.h - 0.5) / self.beta_schedule.get_beta()
        return (v[0] + b - stick, v[1] - b + stick)

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice
        pe = reward - self.q[choice]
        self.q[choice] = min(max(self.q[choice] + p["alpha_c"] * pe, 0.0), 1.0)
        self.q[other] = min(max(self.q[other] + p["alpha_u"] * pe, 0.0), 1.0)
        self.h += p["alpha_h"] * (choice - self.h)
        self.rho += p["alpha_rho"] * (reward - self.rho)
        self.last_choice = choice


class QlearnHierarchical(BasePolicy):
    """
    Two-option hierarchical RL for a 2-armed bandit, with the options as
    hypotheses about which port is better.

    Option 0 starts believing port 0 is better, Q_0 = [0.5 + q_sep, 0.5 - q_sep],
    and option 1 the reverse. A meta-controller learns how well each option
    explains the rewards and mixes their policies:

        p(a) = sum_k softmax(beta_meta * m)[k] * softmax(beta_option * Q_k)[a]

    After each trial, responsibilities r_k = p(option k | chosen action) weight
    the updates:

        Q_k[choice] += alpha_q * r_k * (reward - Q_k[choice])
        m_k         += alpha_meta * r_k * (reward - m_k)

    Between trials (not after a reset) every option's Q-values relax toward
    that option's own starting hypothesis and the meta values toward m_init:
    x <- x0 + tau * (x - x0), so tau = 1 is no forgetting. Because the options
    start apart and forget toward different points, they stay distinct; with
    q_sep = 0 they would be identical forever and the meta level would have no
    effect on choices.

    'logits()' returns log-probabilities, so this policy uses 'NoBeta'.
    'm_init' is fixed at 0.5 (a neutral reward rate) unless enabled.
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_q = ParameterSpec(
            "alpha_q", (0.0, 1.0), description="LR for option Q-values"
        )
        alpha_meta = ParameterSpec(
            "alpha_meta", (0.0, 1.0), description="LR for meta option values"
        )
        tau = ParameterSpec(
            "tau",
            (0.5, 1.0),
            default=1.0,
            description="Retention per trial (1 = no forgetting)",
        )
        q_sep = ParameterSpec(
            "q_sep",
            (0.0, 0.5),
            default=0.25,
            description="How strongly each option starts favouring its port",
        )
        m_init = ParameterSpec(
            "m_init",
            (0.0, 1.0),
            default=0.5,
            active=False,
            description="Initial meta value",
        )
        beta_meta = ParameterSpec(
            "beta_meta", (0.1, 10.0), description="Inverse temp over options"
        )
        beta_option = ParameterSpec(
            "beta_option", (0.1, 10.0), description="Inverse temp within options"
        )

    params: Params

    # Plain Python floats, two options x two ports: these run once per trial in
    # every NLL evaluation, where numpy's per-call overhead dominates.

    def _priors(self):
        d = self.params["q_sep"]
        return ((0.5 + d, 0.5 - d), (0.5 - d, 0.5 + d))

    def reset(self):
        self.q = [list(q0) for q0 in self._priors()]
        m0 = self.params["m_init"]
        self.m = [m0, m0]

    def forget(self):
        tau = self.params["tau"]
        if tau == 1.0:
            return
        for q, q0 in zip(self.q, self._priors()):
            q[0] = q0[0] + tau * (q[0] - q0[0])
            q[1] = q0[1] + tau * (q[1] - q0[1])
        m0 = self.params["m_init"]
        self.m = [m0 + tau * (m - m0) for m in self.m]

    def _mixture(self):
        """(p(option), p(port 1 | option)) for both options."""
        pm1 = 1.0 / (
            1.0 + math.exp(-self.params["beta_meta"] * (self.m[1] - self.m[0]))
        )
        b = self.params["beta_option"]
        p1 = [1.0 / (1.0 + math.exp(-b * (q[1] - q[0]))) for q in self.q]
        return (1.0 - pm1, pm1), p1

    def logits(self):
        pm, p1 = self._mixture()
        p_port1 = pm[0] * p1[0] + pm[1] * p1[1]
        p_port1 = min(max(p_port1, 1e-9), 1.0 - 1e-9)
        return (math.log(1.0 - p_port1), math.log(p_port1))

    def update(self, choice, reward):
        pm, p1 = self._mixture()
        lik = [p1[k] if choice == 1 else 1.0 - p1[k] for k in (0, 1)]
        resp = [pm[0] * lik[0], pm[1] * lik[1]]
        total = resp[0] + resp[1]
        resp = [0.5, 0.5] if total <= 0 else [r / total for r in resp]

        aq = self.params["alpha_q"]
        am = self.params["alpha_meta"]
        for k in (0, 1):
            q = self.q[k]
            q[choice] = min(
                max(q[choice] + aq * resp[k] * (reward - q[choice]), 0.0), 1.0
            )
            self.m[k] += am * resp[k] * (reward - self.m[k])


class QlearnWM(BasePolicy):
    """
    RL + Working Memory model for 2-arm bandit.

    Adapted from Collins & Frank (2012). A model-free RL learner and a
    working memory (WM) system jointly drive action selection. The WM
    system encodes outcomes with a learning rate of 1 (perfect one-shot
    memory) but decays toward chance over time. The mixing weight between
    systems is updated via Bayesian model averaging based on each
    system's predictive accuracy.

    For the 2-arm case (set size n_s = 1), WM capacity is always
    sufficient, so the capacity parameter C is not included.

    Action probabilities:
        p(a) = (1 - w) * softmax(beta_rl * Q_RL) + w * softmax(beta_wm * Q_WM)

    'beta_rl' and 'beta_wm' are internal parameters; the outer softmax
    in 'DecisionModel' should be neutralised by pairing with 'NoBeta()'.

    Reference
    ---------
    Collins, A. G. E. & Frank, M. J. (2012). How much of reinforcement
    learning is working memory, not reinforcement learning? European
    Journal of Neuroscience, 35(7), 1024-1035.
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_rl = ParameterSpec("alpha_rl", (0.0, 1.0), description="RL learning rate")
        beta_rl = ParameterSpec(
            "beta_rl", (0.1, 20.0), description="RL inverse temperature"
        )
        beta_wm = ParameterSpec(
            "beta_wm", (0.1, 20.0), description="WM inverse temperature"
        )
        decay = ParameterSpec(
            "decay", (0.0, 1.0), description="Decay rate toward initial Q"
        )
        w0 = ParameterSpec(
            "w0", (0.0, 1.0), default=0.5, description="Initial WM weight"
        )

    params: Params

    def reset(self):
        self.q_rl = np.full(2, 0.5)
        self.q_wm = np.full(2, 0.5)
        self.w = self.params.get("w0", 0.5)

    def forget(self):
        eps = self.params["decay"]
        self.q_rl += eps * (0.5 - self.q_rl)
        self.q_wm += eps * (0.5 - self.q_wm)

    def logits(self):
        p_rl = _softmax(self.q_rl, self.params["beta_rl"])
        p_wm = _softmax(self.q_wm, self.params["beta_wm"])
        p_mix = (1.0 - self.w) * p_rl + self.w * p_wm
        p_mix = np.clip(p_mix, 1e-9, 1.0)
        return np.log(p_mix)

    def update(self, choice, reward):
        # --- Bayesian update of mixture weight (using pre-update Q) ---
        q_rl_c = np.clip(self.q_rl[choice], 1e-6, 1.0 - 1e-6)
        q_wm_c = np.clip(self.q_wm[choice], 1e-6, 1.0 - 1e-6)

        p_rl_lik = q_rl_c if reward == 1 else (1.0 - q_rl_c)
        p_wm_lik = q_wm_c if reward == 1 else (1.0 - q_wm_c)

        num = p_wm_lik * self.w
        den = num + p_rl_lik * (1.0 - self.w)
        self.w = num / den if den > 1e-12 else 0.5
        self.w = np.clip(self.w, 1e-6, 1.0 - 1e-6)

        # --- RL update (chosen arm only, no counterfactual) ---
        pe = reward - self.q_rl[choice]
        self.q_rl[choice] += self.params["alpha_rl"] * pe
        np.clip(self.q_rl, 0.0, 1.0, out=self.q_rl)

        # --- WM update (perfect one-shot encoding, lr = 1) ---
        self.q_wm[choice] = float(reward)


class QlearnDynamicLR(BasePolicy):
    """
    2-arm Q-learning with dynamic learning rate.

    The learning rate is updated based on the prediction error magnitude.
    """

    class Params(ParameterGroup):
        alpha_c = ParameterSpec(
            "alpha_c", (0.0, 1.0), description="Learning rate for chosen option"
        )
        alpha_u = ParameterSpec(
            "alpha_u", (0.0, 1.0), description="Learning rate for unchosen option"
        )
        w_c = ParameterSpec(
            "w_c",
            (0.0, 1.0),
            description="Weight for chosen option learning rate update",
        )
        w_u = ParameterSpec(
            "w_u",
            (0.0, 1.0),
            description="Weight for unchosen option learning rate update",
        )

    params: Params

    def reset(self):
        self.q = np.full(2, 0.5)
        self.alpha_c = self.params["alpha_c"]
        self.alpha_u = self.params["alpha_u"]

    def forget(self):
        pass

    def logits(self):
        return self.q.copy()

    def update(self, choice, reward):
        other = 1 - choice
        pe = reward - self.q[choice]

        # Update learning rates based on prediction error (EWMA toward |pe|)
        self.alpha_c = (
            self.params["w_c"] * abs(pe) + (1 - self.params["w_c"]) * self.alpha_c
        )
        self.alpha_u = (
            self.params["w_u"] * abs(pe) + (1 - self.params["w_u"]) * self.alpha_u
        )

        # Update Q-values
        self.q[choice] += self.alpha_c * pe
        self.q[other] += self.alpha_u * pe

        # Clamp Q-values to [0, 1]
        np.clip(self.q, 0.0, 1.0, out=self.q)


class QlearnAdaptiveLR(BasePolicy):
    """
    'Qlearn' (6 parameters: counterfactual update, bias, choice trace and
    sticky outside beta) whose learning rates scale with the recent reward
    rate 'r_bar', a running average of rewards from either port.

    With kappa > 0 the learning rates shrink as the reward rate rises: once
    reward is reliable (e.g. settled on the better port), single outcomes
    move the values less; a drop in reward rate brings the learning rates
    back up. kappa < 0 does the opposite. The scaling is multiplicative, so
    a negative alpha_u0 shrinks in magnitude (keeps its sign) as alpha_c0
    does. With kappa_c = kappa_u = 0 this is exactly 'Qlearn'.

    Update rule (r_bar first, so the current outcome sets its own rate):
    r_bar <- r_bar + w_r * (reward - r_bar)
    alpha_c = clip(alpha_c0 * (1 - kappa_c * r_bar), 0, 1)
    alpha_u = clip(alpha_u0 * (1 - kappa_u * r_bar), -1, 1)
    pe = reward - Q[choice]
    Q[choice] += alpha_c * pe
    Q[unchosen] += alpha_u * pe
    h += alpha_h * (choice - h)

    Q clipped to [0, 1]. At each reset Q = [0.5, 0.5], h = 0.5 and
    r_bar = 'r_bar0' (fixed at 0 unless enabled: no reward earned yet in the
    window). Choice values as in 'Qlearn':
    V[0] = beta * (Q[0] + bias) + sticky * (0.5 - h)
    V[1] = beta * (Q[1] - bias) + sticky * (h - 0.5)

    Reward rate is a lagging indicator: right after a reversal it stays high
    for a few trials, so with kappa > 0 the learning rates recover late.
    """

    class Params(ParameterGroup):
        alpha_c0 = ParameterSpec(
            "alpha_c0",
            (0.0, 0.99),
            description="Learning rate for chosen option at zero reward rate",
        )
        alpha_u0 = ParameterSpec(
            "alpha_u0",
            (-0.99, 0.99),
            description="Learning rate for unchosen option at zero reward rate",
        )
        kappa_c = ParameterSpec(
            "kappa_c",
            (-1.0, 1.0),
            default=0.0,
            description="Fractional drop in the chosen learning rate per unit reward rate",
        )
        kappa_u = ParameterSpec(
            "kappa_u",
            (-1.0, 1.0),
            default=0.0,
            description="Fractional drop in the unchosen learning rate per unit reward rate",
        )
        w_r = ParameterSpec(
            "w_r",
            (0.01, 0.5),
            default=0.1,
            description="Learning rate of the reward-rate trace r_bar",
        )
        bias = ParameterSpec(
            "bias", (-1.0, 1.0), default=0.0, description="Bias toward port 0 vs port 1"
        )
        alpha_h = ParameterSpec(
            "alpha_h",
            (0.0, 1.0),
            default=0.0,
            description="Learning rate of the choice (perseverance) trace",
        )
        sticky = ParameterSpec(
            "sticky",
            (0.0, 20.0),
            default=0.0,
            description="Perseverance weight, outside beta (repeat recent choices)",
        )
        r_bar0 = ParameterSpec(
            "r_bar0",
            (0.0, 1.0),
            default=0.0,
            active=False,
            description="Reward rate r_bar at each reset",
        )

    params: Params

    def reset(self):
        self.q = [0.5, 0.5]
        self.h = 0.5
        self.r_bar = self.params["r_bar0"]

    def forget(self):
        pass

    def logits(self):
        # Sticky outside beta, as in 'Qlearn'.
        b = self.params["bias"]
        stick = self.params["sticky"] * (self.h - 0.5) / self.beta_schedule.get_beta()
        return (self.q[0] + b - stick, self.q[1] - b + stick)

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice
        pe = reward - self.q[choice]

        self.r_bar += p["w_r"] * (reward - self.r_bar)
        a_c = min(max(p["alpha_c0"] * (1.0 - p["kappa_c"] * self.r_bar), 0.0), 1.0)
        a_u = min(max(p["alpha_u0"] * (1.0 - p["kappa_u"] * self.r_bar), -1.0), 1.0)

        self.q[choice] = min(max(self.q[choice] + a_c * pe, 0.0), 1.0)
        self.q[other] = min(max(self.q[other] + a_u * pe, 0.0), 1.0)
        self.h += p["alpha_h"] * (choice - self.h)


class QlearnAdaptiveBeta(BasePolicy):
    """
    'Qlearn' (6 parameters: counterfactual update, bias, choice trace and
    sticky outside beta) whose whole decision is scaled by a gain set by the
    recent reward rate 'r_bar', a running average of rewards from either port.

    The gain multiplies the value and the perseveration terms alike, so it
    acts as a temperature on the whole choice: with kappa_beta > 0 a low
    reward rate pulls choices toward 50/50 (exploration, perseveration
    included) and a high reward rate sharpens them (exploitation);
    kappa_beta < 0 does the opposite. 'beta' and 'sticky' are their values at
    r_bar = 0.5. (Scaling the value term alone would instead hand the choice
    to perseveration when the reward rate is low.) With kappa_beta = 0 this
    is exactly 'Qlearn'.

    Update rule (learning rates fixed, as in 'Qlearn'):
    pe = reward - Q[choice]
    Q[choice] += alpha_c * pe
    Q[unchosen] += alpha_u * pe
    h += alpha_h * (choice - h)
    r_bar <- r_bar + w_r * (reward - r_bar)

    Q clipped to [0, 1]. At each reset Q = [0.5, 0.5], h = 0.5 and
    r_bar = 'r_bar0' (fixed at 0 unless enabled: no reward earned yet in the
    window, so with kappa_beta > 0 each window starts exploratory and the
    gain rises as rewards accumulate). Choice values, from the reward rate
    after the last trial:
    g = exp(kappa_beta * (r_bar - 0.5))
    V[0] = g * (beta * (Q[0] + bias) + sticky * (0.5 - h))
    V[1] = g * (beta * (Q[1] - bias) + sticky * (h - 0.5))
    """

    class Params(ParameterGroup):
        alpha_c = ParameterSpec(
            "alpha_c", (0.0, 0.99), description="Learning rate for chosen option"
        )
        alpha_u = ParameterSpec(
            "alpha_u", (-0.99, 0.99), description="Learning rate for unchosen option"
        )
        bias = ParameterSpec(
            "bias", (-1.0, 1.0), default=0.0, description="Bias toward port 0 vs port 1"
        )
        alpha_h = ParameterSpec(
            "alpha_h",
            (0.0, 1.0),
            default=0.0,
            description="Learning rate of the choice (perseverance) trace",
        )
        sticky = ParameterSpec(
            "sticky",
            (0.0, 20.0),
            default=0.0,
            description="Perseverance weight, outside beta (repeat recent choices)",
        )
        w_r = ParameterSpec(
            "w_r",
            (0.01, 0.5),
            default=0.1,
            description="Learning rate of the reward-rate trace r_bar",
        )
        kappa_beta = ParameterSpec(
            "kappa_beta",
            (-3.0, 3.0),
            default=0.0,
            description="Log change in the decision gain per unit reward rate (gain 1 at r_bar = 0.5)",
        )
        r_bar0 = ParameterSpec(
            "r_bar0",
            (0.0, 1.0),
            default=0.0,
            active=False,
            description="Reward rate r_bar at each reset",
        )

    params: Params

    def reset(self):
        self.q = [0.5, 0.5]
        self.h = 0.5
        self.r_bar = self.params["r_bar0"]

    def forget(self):
        pass

    def logits(self):
        # Sticky outside beta, as in 'Qlearn'; the gain scales both terms (the
        # caller multiplies by beta, so g * logits gives g * (beta * values + sticky term)).
        b = self.params["bias"]
        g = math.exp(self.params["kappa_beta"] * (self.r_bar - 0.5))
        stick = self.params["sticky"] * (self.h - 0.5) / self.beta_schedule.get_beta()
        return (g * (self.q[0] + b - stick), g * (self.q[1] - b + stick))

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice
        pe = reward - self.q[choice]
        self.q[choice] = min(max(self.q[choice] + p["alpha_c"] * pe, 0.0), 1.0)
        self.q[other] = min(max(self.q[other] + p["alpha_u"] * pe, 0.0), 1.0)
        self.h += p["alpha_h"] * (choice - self.h)
        self.r_bar += p["w_r"] * (reward - self.r_bar)


class QlearnDynamicSticky(BasePolicy):
    """
    'Qlearn' (6 parameters: counterfactual update, bias, choice trace and
    sticky outside beta) whose perseverance weight changes along the window,
    on a fixed schedule of trials since the reset (not of outcomes):

    sticky_t = sticky * (1 - (1 - k0) * exp(-t / tau))

    t counts the trials already made in the window (0 for the first choice).
    'k0' is the weight at the window start as a fraction of 'sticky', the
    weight late in the window, and 'tau' (trials) how fast it gets there.
    k0 < 1: perseverance builds up through the window (e.g. satiation);
    k0 > 1: it fades. With k0 = 1 this is exactly 'Qlearn' (tau then has no
    effect).

    Update rule, as in 'Qlearn':
    pe = reward - Q[choice]
    Q[choice] += alpha_c * pe
    Q[unchosen] += alpha_u * pe
    h += alpha_h * (choice - h)

    Q clipped to [0, 1]. At each reset Q = [0.5, 0.5], h = 0.5 and t = 0.
    Choice values:
    V[0] = beta * (Q[0] + bias) + sticky_t * (0.5 - h)
    V[1] = beta * (Q[1] - bias) + sticky_t * (h - 0.5)
    """

    class Params(ParameterGroup):
        alpha_c = ParameterSpec(
            "alpha_c", (0.0, 0.99), description="Learning rate for chosen option"
        )
        alpha_u = ParameterSpec(
            "alpha_u", (-0.99, 0.99), description="Learning rate for unchosen option"
        )
        bias = ParameterSpec(
            "bias", (-1.0, 1.0), default=0.0, description="Bias toward port 0 vs port 1"
        )
        alpha_h = ParameterSpec(
            "alpha_h",
            (0.0, 1.0),
            default=0.0,
            description="Learning rate of the choice (perseverance) trace",
        )
        sticky = ParameterSpec(
            "sticky",
            (0.0, 20.0),
            default=0.0,
            description="Perseverance weight late in the window, outside beta",
        )
        k0 = ParameterSpec(
            "k0",
            (0.0, 3.0),
            default=1.0,
            description="Perseverance weight at the window start, as a fraction of sticky",
        )
        tau = ParameterSpec(
            "tau",
            (1.0, 1000.0),
            default=100.0,
            description="Trials for the perseverance weight to move from k0 * sticky to sticky (time constant)",
        )

    params: Params

    def reset(self):
        self.q = [0.5, 0.5]
        self.h = 0.5
        self.t = 0

    def forget(self):
        pass

    def logits(self):
        # Sticky outside beta, as in 'Qlearn' (the caller multiplies by beta).
        p = self.params
        k = 1.0 - (1.0 - p["k0"]) * math.exp(-self.t / p["tau"])
        stick = k * p["sticky"] * (self.h - 0.5) / self.beta_schedule.get_beta()
        b = p["bias"]
        return (self.q[0] + b - stick, self.q[1] - b + stick)

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice
        pe = reward - self.q[choice]
        self.q[choice] = min(max(self.q[choice] + p["alpha_c"] * pe, 0.0), 1.0)
        self.q[other] = min(max(self.q[other] + p["alpha_u"] * pe, 0.0), 1.0)
        self.h += p["alpha_h"] * (choice - self.h)
        self.t += 1


class QlearnDiff(BasePolicy):
    """
    2-arm Q-learning driven by the chosen-vs-unchosen value difference.

    Instead of separate learning rates for the chosen and unchosen options,
    a single alpha scales the update to the difference between them. The
    difference ranges from -1 to 1, and the reward prediction error compares
    reward to the absolute value of that difference (how confidently the
    chosen option was already favoured).

    Update rule:
    diff = Q[choice] - Q[unchosen]
    rpe = reward - abs(diff)
    Q[choice] += alpha * rpe
    Q[unchosen] -= alpha * rpe

    Choice logits:
    logit[0] = Q[0] + bias
    logit[1] = Q[1] - bias
    """

    class Params(ParameterGroup):
        alpha = ParameterSpec(
            "alpha",
            (0.0, 0.99),
            description="Learning rate for the chosen/unchosen difference",
        )

        bias = ParameterSpec(
            "bias", (-1.0, 1.0), default=0.0, description="Bias toward port 0 vs port 1"
        )

    params: Params

    def reset(self):
        self.q = np.full(2, 0.5)

    def forget(self):
        pass

    def logits(self):
        b = self.params["bias"]
        return np.array([self.q[0] + b, self.q[1] - b])

    def update(self, choice, reward):
        a = self.params["alpha"]
        other = 1 - choice

        diff = self.q[choice] - self.q[other]
        rpe = reward - abs(diff)

        delta = a * rpe
        self.q[choice] += delta
        self.q[other] -= delta

        self.q[:] = np.clip(self.q, 0.0, 1.0)
