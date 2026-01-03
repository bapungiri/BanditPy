import numpy as np
from banditpy.models.policy.base import BasePolicy, ParameterSpec


class Qlearn2Arm(BasePolicy):
    """
    Vanilla 2-arm Q-learning with counterfactual updates.

    Update rule:
    Q[choice] += alpha_c * (reward - Q[choice])
    Q[unchosen] += alpha_u * (reward - Q[choice])
    """

    parameters = [
        ParameterSpec(
            "alpha_c", (0.0, 1.0), description="Learning rate for chosen option"
        ),
        ParameterSpec(
            "alpha_u", (0.0, 1.0), description="Learning rate for unchosen option"
        ),
        ParameterSpec(
            "beta", (0.1, 20.0), description="Inverse temperature for softmax"
        ),
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


class QlearnH2Arm(BasePolicy):

    parameters = [
        ParameterSpec("alpha_c", (-1.0, 1.0), description="Learning rate (chosen)"),
        ParameterSpec("alpha_u", (-1.0, 1.0), description="Learning rate (unchosen)"),
        ParameterSpec("alpha_h", (0.0, 1.0), description="Perseverance learning"),
        ParameterSpec("scaler", (1, 10.0), description="Perseverance scale"),
        ParameterSpec("beta", (0.01, 20.0), description="Inverse temperature"),
    ]

    def reset(self):
        self.q0 = 0.5
        self.q = np.array([self.q0, self.q0], dtype=float)
        self.h = 0.5

        p = self.params
        self._ac = p["alpha_c"]
        self._au = p["alpha_u"]
        self._ah = p["alpha_h"]
        self._sc = p["scaler"]

    def forget(self):
        return

    def logits(self):
        h = self.h
        bias0 = h - 0.5
        bias1 = 0.5 - h
        return np.array(
            (
                self.q[0] + self.params["scaler"] * bias0,
                self.q[1] + self.params["scaler"] * bias1,
            ),
            dtype=float,
        )

    def update(self, choice, reward):
        p = self.params
        other = 1 - choice

        rpe = reward - self.q[choice]

        self.q[choice] += p["alpha_c"] * rpe
        self.q[other] += p["alpha_u"] * rpe

        # fast in-place clamp
        np.minimum(self.q, 1.0, out=self.q)
        np.maximum(self.q, 0.0, out=self.q)

        self.h += p["alpha_h"] * (choice - self.h)
