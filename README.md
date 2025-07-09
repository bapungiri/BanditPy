# BanditPy

BanditPy provides repository to analyze and model bandit tasks.

## Functionalities

### Core classes
- Bandit2Arm
- 

### Models
- Qlearn2Arm
- Logistic2Arm
- BandiTrainer2Arm
- Thompson2Arm

### Analysis
- SwitchProb2Arm


## Dependencies
- Python 3.13



## Example

```
import numpy as np
import pandas as pd
from banditpy.models import Qlearn2Arm
from banditpy.core import Bandit2Arm
import mab_subjects

true_probs = [0.3, 0.7]
n_trials = 200
choices, rewards = [], []

for _ in range(n_trials):
    choice = np.random.choice([0, 1], p=[0.4, 0.6])  # biased animal
    reward = np.random.rand() < true_probs[choice]
    choices.append(choice)
    rewards.append(reward)


task = Bandit2Arm(probs=probs,choices=choices,rewards=rewards) 
qlearn = QlearningEstimator(task)
qlearn.fit(
    x0=None,
    bounds=np.array([(-1, 1), (-1, 1), (0.005, 20)]),
    method="diff_evolution",
    n_opts=5,
    n_cpu=4,
)

qlearn.print_params()

```
