import math

from .base import BasePolicy, ParameterGroup, ParameterSpec
from .beta_schedule import NoBeta


class MVTForager(BasePolicy):
    """
    Stay/leave foraging model after the marginal value theorem (Charnov 1976), with the
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

    Optional perseverance (off by default; 'sticky' = 0 is the model above exactly):
    a choice trace h of port 0, as in Qlearn, h <- h + alpha_h * (1[choice = 0] - h)
    after every trial, and the trace of the port chosen last, h_cur (h or 1 - h),
    lowers the switch logit:

        x = beta * (rho - r_hat) + gamma - sticky * (h_cur - 0.5)

    Unlike the constant 'gamma', this grows with the time spent on a port and
    carries across switches (returning to a recently used port is easier).
    h resets to 0.5.

    'gamma' is the baseline switching (negative = sticky). At each reset
    rho = 'rho0' and r_hat = 'r0', and the first choice is 50/50 (or set by the
    optional 'bias' toward port 0). With 'alpha_rho' = 0, rho stays at rho0 and
    'beta * rho0 + gamma' can't be separated, so fix one of them in that case.
    'logits()' returns log-probabilities, so this policy uses 'NoBeta'.
    No Q-values: see 'QlearnMVTHybrid' (qlearn.py) for Qlearn with rho in the
    value of the other port.
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
            (0.1, 30.0),
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
        alpha_h = ParameterSpec(
            "alpha_h",
            (0.0, 1.0),
            default=0.0,
            active=False,
            description="Learning rate of the choice (perseverance) trace",
        )
        sticky = ParameterSpec(
            "sticky",
            (0.0, 20.0),
            default=0.0,
            active=False,
            description="Perseverance weight on the switch logit (stay where the trace is high)",
        )

    params: Params

    def reset(self):
        self.current = None  # port chosen on the last trial
        self.r_hat = self.params["r0"]
        self.rho = self.params["rho0"]
        self.h = 0.5  # choice trace of port 0

    def forget(self):
        pass

    def demand(self):
        """What the current port must pay to be worth staying (here rho)."""
        return self.rho

    def gap_logit(self):
        """Switch logit from the opportunity-cost gap, before the choice trace."""
        p = self.params
        return p["beta"] * (self.demand() - self.r_hat) + p["gamma"]

    def logits(self):
        p = self.params
        side = 2.0 * p["bias"]  # toward port 0
        if self.current is None:
            p0 = 1.0 / (1.0 + math.exp(-side))
        else:
            x = self.gap_logit()
            if p["sticky"]:
                h_cur = self.h if self.current == 0 else 1.0 - self.h
                x -= p["sticky"] * (h_cur - 0.5)
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
        self.h += p["alpha_h"] * ((1.0 if choice == 0 else 0.0) - self.h)
        self.current = choice


class MVTForagerKappa(MVTForager):
    """
    'MVTForager' with separate weights on the environment rate and the local rate:

        x = kappa * rho - beta * r_hat + gamma [- sticky * (h_cur - 0.5)]
        P(switch) = sigmoid(x)

    rho, r_hat, the optional choice trace and 'bias' are exactly as in
    'MVTForager'. 'kappa' = 'beta' is 'MVTForager' (the gap beta * (rho - r_hat));
    'kappa' = 0 drops the environment rate. With the weight tied to beta, a fast
    rho (following the current port's rewards) cancels r_hat, so 'alpha_rho' had
    to stay slow to keep rho's effect small; a free 'kappa' lets 'alpha_rho' set
    only the timescale. 'kappa' may be negative (a rich environment lowering
    switching). One parameter more than 'MVTForager'. With 'alpha_rho' = 0,
    'kappa * rho0' and 'gamma' can't be separated.
    """

    class Params(ParameterGroup):
        alpha_l = MVTForager.Params.alpha_l
        alpha_rho = MVTForager.Params.alpha_rho
        beta = ParameterSpec(
            "beta",
            (0.1, 30.0),
            description="Weight of the local rate r_hat (switching falls as it rises)",
        )
        kappa = ParameterSpec(
            "kappa",
            (-30.0, 30.0),
            default=0.0,
            description="Weight of the environment rate rho (kappa = beta is MVTForager)",
        )
        gamma = MVTForager.Params.gamma
        r0 = MVTForager.Params.r0
        rho0 = MVTForager.Params.rho0
        bias = MVTForager.Params.bias
        alpha_h = MVTForager.Params.alpha_h
        sticky = MVTForager.Params.sticky

    params: Params

    def gap_logit(self):
        p = self.params
        return p["kappa"] * self.demand() - p["beta"] * self.r_hat + p["gamma"]


class MVTForagerThirst(MVTForager):
    """
    'MVTForager' (learned environment rate rho) plus a thirst term that raises
    the demand at each window start and fades as the animal drinks:

        demand_n = rho + delta * exp(-n / tau_s)
        x = beta * (demand_n - r_hat) + gamma [- sticky * (h_cur - 0.5)]
        P(switch) = sigmoid(x)

    rho, r_hat, the optional choice trace and 'bias' are exactly as in
    'MVTForager'. 'n' is the window's clock, chosen at construction:
    clock="rewards" (default, rewards earned so far in the window) or
    clock="trials". n resets to 0 at each window start.

    'delta' = 0 is 'MVTForager'; 'alpha_rho' = 0 is 'MVTForagerSatiety'
    (rho0 + delta * exp(-n / tau_s), with rho0 absorbed by 'gamma'). Two
    parameters more than 'MVTForager'. 'rho0' (which already starts above the
    rate the animal earns) and 'delta' can trade off: check recovery.
    """

    class Params(ParameterGroup):
        alpha_l = MVTForager.Params.alpha_l
        alpha_rho = MVTForager.Params.alpha_rho
        beta = MVTForager.Params.beta
        gamma = MVTForager.Params.gamma
        r0 = MVTForager.Params.r0
        rho0 = MVTForager.Params.rho0
        delta = ParameterSpec(
            "delta",
            (0.0, 1.0),
            default=0.0,
            description="Extra demand from thirst at each window start (fades with n)",
        )
        tau_s = ParameterSpec(
            "tau_s",
            (1.0, 2000.0),
            default=100.0,
            description="Time constant of the thirst term, in units of the clock (rewards or trials)",
        )
        bias = MVTForager.Params.bias
        alpha_h = MVTForager.Params.alpha_h
        sticky = MVTForager.Params.sticky

    params: Params

    def __init__(self, clock="rewards", **kwargs):
        if clock not in ("rewards", "trials"):
            raise ValueError("clock must be 'rewards' or 'trials'")
        self.clock = clock
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.n = 0.0  # window clock: rewards or trials so far

    def thirst(self):
        """Fraction of the window-start thirst left, exp(-n / tau_s)."""
        return math.exp(-self.n / self.params["tau_s"])

    def demand(self):
        return self.rho + self.params["delta"] * self.thirst()

    def update(self, choice, reward):
        super().update(choice, reward)
        self.n += reward if self.clock == "rewards" else 1.0


class MVTForagerSatiety(BasePolicy):
    """
    'MVTForager' whose environment rate rho is not learned as a running average
    but falls exponentially along the window, set by its initial value and a
    time constant: the animal starts each window thirsty and demanding (a port
    must pay a lot to be worth staying) and becomes more tolerant of misses as
    it drinks.

        rho_n = rho0 * exp(-n / tau_rho)

    'n' is the window's clock, chosen at construction:

    - clock="rewards" (default): rewards earned so far in the window (water drunk);
    - clock="trials": trials made so far in the window.

    n resets to 0 at each window start and counts up to the last trial (the
    current choice uses rho from the trials before it). With the rewards clock,
    rho stays high while rewards are scarce (e.g. a low-low first block) and
    falls fast when they are plentiful.

    A constant added to rho only shifts the switch logit by beta * constant,
    which 'gamma' absorbs, so decaying to 0 is the same model as decaying to any
    fixed level; 'rho_inf' (inactive, 0) can be enabled to set that level:
    rho_n = rho_inf + (rho0 - rho_inf) * exp(-n / tau_rho). rho0 = rho_inf (or
    tau_rho -> inf) is a fixed rho, i.e. 'MVTForager' with alpha_rho = 0.
    'rho0' and 'tau_rho' replace MVTForager's 'rho0' and 'alpha_rho' (same
    number of parameters).

    Local rate and switch decision as in 'MVTForager':

        r_hat <- r_hat + alpha_l * (reward - r_hat)     after staying
        r_hat <- r0    + alpha_l * (reward - r0)        after switching
        x = beta * (rho_n - r_hat) + gamma [- sticky * (h_cur - 0.5)]
        P(switch) = sigmoid(x)

    The optional choice trace ('alpha_h', 'sticky', off by default) and port
    'bias' are as in 'MVTForager'. 'logits()' returns log-probabilities,
    so this policy uses 'NoBeta'.
    """

    default_beta_schedule = NoBeta

    class Params(ParameterGroup):
        alpha_l = ParameterSpec(
            "alpha_l",
            (0.0, 1.0),
            description="Learning rate of the current port's local rate",
        )
        beta = ParameterSpec(
            "beta",
            (0.1, 30.0),
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
            description="Environment rate rho at each window start",
        )
        tau_rho = ParameterSpec(
            "tau_rho",
            (1.0, 2000.0),
            default=100.0,
            description="Time constant of rho's decay, in units of the clock (rewards or trials)",
        )
        rho_inf = ParameterSpec(
            "rho_inf",
            (0.0, 1.0),
            default=0.0,
            active=False,
            description="Level rho decays to (not separable from gamma, so fixed at 0)",
        )
        bias = ParameterSpec(
            "bias",
            (-1.0, 1.0),
            default=0.0,
            active=False,
            description="Bias toward port 0 vs port 1 (adds +-2*bias to the port-0 logit)",
        )
        alpha_h = ParameterSpec(
            "alpha_h",
            (0.0, 1.0),
            default=0.0,
            active=False,
            description="Learning rate of the choice (perseverance) trace",
        )
        sticky = ParameterSpec(
            "sticky",
            (0.0, 20.0),
            default=0.0,
            active=False,
            description="Perseverance weight on the switch logit (stay where the trace is high)",
        )

    params: Params

    def __init__(self, clock="rewards", **kwargs):
        if clock not in ("rewards", "trials"):
            raise ValueError("clock must be 'rewards' or 'trials'")
        self.clock = clock
        super().__init__(**kwargs)

    def reset(self):
        self.current = None  # port chosen on the last trial
        self.r_hat = self.params["r0"]
        self.n = 0.0  # window clock: rewards or trials so far
        self.h = 0.5  # choice trace of port 0

    def forget(self):
        pass

    def rho(self):
        p = self.params
        return p["rho_inf"] + (p["rho0"] - p["rho_inf"]) * math.exp(-self.n / p["tau_rho"])

    def logits(self):
        p = self.params
        side = 2.0 * p["bias"]  # toward port 0
        if self.current is None:
            p0 = 1.0 / (1.0 + math.exp(-side))
        else:
            x = p["beta"] * (self.rho() - self.r_hat) + p["gamma"]
            if p["sticky"]:
                h_cur = self.h if self.current == 0 else 1.0 - self.h
                x -= p["sticky"] * (h_cur - 0.5)
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
        self.n += reward if self.clock == "rewards" else 1.0
        self.h += p["alpha_h"] * ((1.0 if choice == 0 else 0.0) - self.h)
        self.current = choice
