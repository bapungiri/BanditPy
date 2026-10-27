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

    Between trials (not at a reset), the value of the arm not chosen on the
    previous trial decays toward its initial 0.5:
    Q[unchosen] += decay * (0.5 - Q[unchosen])

    Choice logits:
    logit[0] = Q[0] + bias + sticky * (0.5 - h)
    logit[1] = Q[1] - bias + sticky * (h - 0.5)

    'sticky' >= 0 favours repeating recent choices (perseverance only; no
    alternation). It is allowed above 1, where perseverance can outweigh
    any Q-value difference: fitted animals sat at a +1 bound.
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
    choice trace.
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
            (0.0, 10.0),
            default=0.0,
            active=True,
            description="Perseverance weight (repeat recent choices)",
        )
        decay = ParameterSpec(
            "decay",
            (0.0, 1.0),
            default=0.0,
            active=False,
            description="Decay of the unchosen Q-value toward 0.5 per trial",
        )

    params: Params

    # Plain Python floats instead of 2-element numpy arrays: these run once
    # per trial in every NLL evaluation, where numpy's per-call overhead
    # (np.clip, array construction) dominates the arithmetic.

    def reset(self):
        self.q = [0.5, 0.5]
        self.h = 0.5
        self.last_choice = None

    def forget(self):
        d = self.params["decay"]
        if d and self.last_choice is not None:
            other = 1 - self.last_choice
            self.q[other] += d * (0.5 - self.q[other])

    def logits(self):
        b = self.params["bias"]
        stick = self.params["sticky"] * (self.h - 0.5)
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
        Q[unchosen] += alpha_u * delta          (both clipped to [-1, 1])
        h           += alpha_h * (choice - h)

    Choice, as in 'Qlearn':

        logit[0] = Q[0] + bias + sticky * (0.5 - h)
        logit[1] = Q[1] - bias + sticky * (h - 0.5)

    Q now means better or worse than the environment's average, so values can
    be negative: they are clipped to [-1, 1], not [0, 1]. At each reset
    Q = [0.5, 0.5], h = 0.5 and rho = 'rho0' (fixed at 0 unless enabled), so
    with alpha_rho = 0 this is exactly 'Qlearn' (QlearnSticky): the extra
    parameter alpha_rho tests whether a learned reward rate helps. Because rho shifts both ports' values by the same amount,
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
            (0.0, 10.0),
            default=0.0,
            description="Perseverance weight (repeat recent choices)",
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
        b = self.params["bias"]
        stick = self.params["sticky"] * (self.h - 0.5)
        return (self.q[0] + b - stick, self.q[1] - b + stick)

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice
        delta = (reward - self.rho) - self.q[choice]
        self.q[choice] = min(max(self.q[choice] + p["alpha_c"] * delta, -1.0), 1.0)
        self.q[other] = min(max(self.q[other] + p["alpha_u"] * delta, -1.0), 1.0)
        self.h += p["alpha_h"] * (choice - self.h)
        self.rho += p["alpha_rho"] * (reward - self.rho)


class QlearnMVT(BasePolicy):
    """
    Stay/leave model after the marginal value theorem (Charnov 1976), with the
    opportunity cost learned from rewards (Constantino & Daw 2015). The current
    port is a patch: stay while its local reward rate beats the environment's
    rate 'rho', leave when it falls below.

    Local rate of the port chosen on the last trial (fast; restarts at 'r0'
    when the animal switches, a new patch starts fresh):

        r_hat <- r_hat + alpha_l * (reward - r_hat)     after staying
        r_hat <- r0    + alpha_l * (reward - r0)        after switching

    Environment rate, the opportunity cost (slow, every trial):

        rho <- rho + alpha_rho * (reward - rho)

    Switch decision and choice:

        P(switch) = sigmoid(beta * (rho - r_hat) + gamma)
        P(c_t = c_{t-1}) = 1 - P(switch)

    'gamma' is the baseline switching (negative = sticky). At each reset
    rho = 'rho0' and r_hat = 'r0', and the first choice is 50/50 (or set by the
    optional 'bias' toward port 0). With 'alpha_rho' = 0, rho stays at rho0 and
    'beta * rho0 + gamma' can't be separated, so fix one of them in that case.
    'logits()' returns log-probabilities, so this policy uses 'NoBeta'.
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_l = ParameterSpec(
            "alpha_l",
            (0.0, 1.0),
            description="Learning rate of the current port's local rate",
        )
        alpha_rho = ParameterSpec(
            "alpha_rho",
            (0.0, 1.0),
            description="Learning rate of the environment rate rho",
        )
        beta = ParameterSpec(
            "beta",
            (0.1, 50.0),
            description="Sensitivity to the opportunity-cost gap rho - r_hat",
        )
        gamma = ParameterSpec(
            "gamma",
            (-10.0, 10.0),
            default=0.0,
            description="Baseline switching (negative = sticky)",
        )
        r0 = ParameterSpec(
            "r0",
            (0.0, 1.0),
            default=0.5,
            description="Estimate a newly visited port starts from",
        )
        rho0 = ParameterSpec(
            "rho0",
            (0.0, 1.0),
            default=0.5,
            description="Environment rate rho at each reset",
        )
        bias = ParameterSpec(
            "bias",
            (-1.0, 1.0),
            default=0.0,
            active=False,
            description="Bias toward port 0 vs port 1 (adds +-2*bias to the port-0 logit)",
        )

    params: Params

    def reset(self):
        self.current = None  # port chosen on the last trial
        self.r_hat = self.params["r0"]
        self.rho = self.params["rho0"]

    def forget(self):
        pass

    def logits(self):
        p = self.params
        side = 2.0 * p["bias"]  # toward port 0
        if self.current is None:
            p0 = 1.0 / (1.0 + math.exp(-side))
        else:
            x = p["beta"] * (self.rho - self.r_hat) + p["gamma"]
            p_switch = (
                1.0 / (1.0 + math.exp(-x))
                if x >= 0
                else math.exp(x) / (1.0 + math.exp(x))
            )
            p_stay = 1.0 - p_switch
            p0 = p_stay if self.current == 0 else p_switch
            if side:  # side bias on top of the stay/leave rule
                lo = math.log(max(p0, 1e-12)) - math.log(max(1.0 - p0, 1e-12)) + side
                p0 = 1.0 / (1.0 + math.exp(-lo))
        p0 = min(max(p0, 1e-9), 1.0 - 1e-9)
        return (math.log(p0), math.log(1.0 - p0))

    def update(self, choice, reward):
        p = self.params
        if choice != self.current:
            self.r_hat = p["r0"]  # new patch
        self.r_hat += p["alpha_l"] * (reward - self.r_hat)
        self.rho += p["alpha_rho"] * (reward - self.rho)
        self.current = choice


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
    2-arm Q-learning with a reward-rate dependent adaptive learning rate.

    The learning rate decreases as the recent (EWMA) reward rate rises,
    capturing the idea that once reward has become reliably predictable
    (e.g. the animal has settled on the better port), further updates
    should shrink; a drop in reward rate raises the learning rate back up.
    This is a simple scalar heuristic, not a reimplementation of any
    specific published model.

    Note that reward rate is a lagging indicator: right after a reversal
    it stays high for a few trials before dropping, so the learning rate
    is briefly slow to recover exactly when fast relearning matters most.

    Update rule:
    r_bar <- r_bar + w_r * (reward - r_bar)
    alpha_c <- clip(alpha_c0 - kappa_c * r_bar, 0, 1)
    alpha_u <- clip(alpha_u0 - kappa_u * r_bar, 0, 1)
    pe = reward - Q[choice]
    Q[choice] += alpha_c * pe
    Q[unchosen] += alpha_u * pe
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
            (0.0, 2.0),
            description="Drop in chosen learning rate per unit reward rate",
        )
        kappa_u = ParameterSpec(
            "kappa_u",
            (0.0, 2.0),
            description="Drop in unchosen learning rate per unit reward rate",
        )
        w_r = ParameterSpec(
            "w_r",
            (0.02, 0.5),
            description="EWMA weight for the reward-rate trace",
        )
        bias = ParameterSpec(
            "bias", (-1.0, 1.0), default=0.0, description="Bias toward port 0 vs port 1"
        )

    params: Params

    def reset(self):
        self.q = np.full(2, 0.5)
        self.r_bar = 0.5
        self.alpha_c = self.params["alpha_c0"]
        self.alpha_u = self.params["alpha_u0"]

    def forget(self):
        pass

    def logits(self):
        b = self.params["bias"]
        return np.array([self.q[0] + b, self.q[1] - b])

    def update(self, choice, reward):
        other = 1 - choice
        pe = reward - self.q[choice]

        self.r_bar += self.params["w_r"] * (reward - self.r_bar)

        self.alpha_c = np.clip(
            self.params["alpha_c0"] - self.params["kappa_c"] * self.r_bar, 0.0, 1.0
        )
        self.alpha_u = np.clip(
            self.params["alpha_u0"] - self.params["kappa_u"] * self.r_bar, -1.0, 1.0
        )

        self.q[choice] += self.alpha_c * pe
        self.q[other] += self.alpha_u * pe
        np.clip(self.q, 0.0, 1.0, out=self.q)


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
