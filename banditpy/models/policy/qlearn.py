import numpy as np
from banditpy.models.policy.base import BasePolicy, ParameterSpec


class QLearn(BasePolicy):
    """
    Vanilla 2-arm Q-learning with counterfactual updates.
    """

    parameters = [
        ParameterSpec("alpha_c", (0.0, 1.0)),
        ParameterSpec("alpha_u", (0.0, 1.0)),
        ParameterSpec("beta", (0.1, 20.0)),
    ]

    def reset(self):
        self.q = np.full(2, 0.5)

    def forget(self):
        pass  # no forgetting in vanilla Q-learning

    def logits(self):
        return self.q.copy()

    def update(self, choice, reward):
        a_c = self.params["alpha_c"]
        a_u = self.params["alpha_u"]

        other = 1 - choice
        pe = reward - self.q[choice]

        self.q[choice] += a_c * pe
        self.q[other] += a_u * pe

        self.q[:] = np.clip(self.q, 0.0, 1.0)


class QLearnPerseverance(BasePolicy):
    """
    Q-learning with perseverance / choice history bias.
    """

    parameters = [
        ParameterSpec("alpha_c", (0.0, 1.0)),
        ParameterSpec("alpha_u", (0.0, 1.0)),
        ParameterSpec("alpha_h", (0.0, 1.0)),
        ParameterSpec("scaler", (0.0, 10.0)),
        ParameterSpec("beta", (0.1, 20.0)),
    ]

    def reset(self):
        self.q = np.full(2, 0.5)
        self.h = 1.5  # midpoint between choices 1 and 2

    def forget(self):
        pass

    def logits(self):
        # bias term added inside logits
        bias = np.array([self.h - 1.5, 1.5 - self.h])
        return self.q + self.params["scaler"] * bias

    def update(self, choice, reward):
        p = self.params

        other = 1 - choice
        pe = reward - self.q[choice]

        self.q[choice] += p["alpha_c"] * pe
        self.q[other] += p["alpha_u"] * pe
        self.q[:] = np.clip(self.q, 0.0, 1.0)

        # perseverance update (choice is 0/1 → map to 1/2)
        self.h += p["alpha_h"] * ((choice + 1) - self.h)
