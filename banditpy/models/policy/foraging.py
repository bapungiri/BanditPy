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
