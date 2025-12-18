# BanditPy

BanditPy is a toolkit for parsing, analyzing, and modeling behavior in two-armed bandit experiments. It bundles data ingestion utilities, task containers, and multiple decision-making models that can be mixed and matched for comparative studies.

## Features
- Structured data containers in [banditpy/core/mab.py](banditpy/core/mab.py) with session, block, and window metadata, filtering helpers, and pandas export.
- Turnkey `.dat` log loader in [banditpy/io/datio.py](banditpy/io/datio.py) that reconstructs trials, timestamps, and reward probabilities from raw hardware exports.
- Model zoo covering Q-learning, logistic regression, Thompson sampling, and UCB variants inside [banditpy/models](banditpy/models) with likelihood-based fitting and diagnostic helpers.
- Ready-to-use behavioral metrics such as switching analyses in [banditpy/analyses/switch_probability.py](banditpy/analyses/switch_probability.py).
- Plotting and utility helpers in [banditpy/plots](banditpy/plots) and [banditpy/utils](banditpy/utils) to streamline exploratory workflows.

## Installation
1. Clone the repository and move into it.
2. Create and activate a Python 3.10+ environment.
3. Install in editable mode and add the scientific stack:

```
pip install -e .
pip install numpy pandas scipy scikit-learn
```

## Quickstart

```python
from pathlib import Path

from banditpy.io.datio import dat2ArmIO
from banditpy.models.qlearn import Qlearn2Arm
from banditpy.models.ucb import UCB2Arm
from banditpy.models.thompson_models import Thompson2Arm
from banditpy.analyses.switch_probability import SwitchProb2Arm

# Load and lightly filter a session
task = dat2ArmIO(Path("/path/to/dat_logs"))
task = task.filter_by_trials(min_trials=50)

# Fit a vanilla Q-learning model
q_model = Qlearn2Arm(task, model="vanilla")
q_model.fit(
    bounds=[(1e-3, 1.0), (1e-3, 1.0), (0.1, 15.0)],
    n_optimize=10,
)
q_model.print_params()

# Compare an exploration-based alternative
ucb_model = UCB2Arm(task, mode="bayesian")
ucb_model.fit(n_starts=5)
ucb_model.print_params()

# Inspect posterior dynamics under Thompson sampling
th_model = Thompson2Arm(task, lr_mode="shared", use_analytic=False)
th_model.fit(n_starts=5)
alpha_traj, beta_traj, means = th_model.simulate_posteriors()

# Basic switching statistics
switch_prob = SwitchProb2Arm(task).by_session()
print(f"Mean switch probability: {switch_prob:.3f}")
```

## Module Guide
- [banditpy/core](banditpy/core) – DataManager and BanditTask abstractions with metadata handling and segmentation utilities.
- [banditpy/io](banditpy/io) – Parsers for raw behavior logs and helpers for building Bandit2Arm objects from disk.
- [banditpy/models/qlearn.py](banditpy/models/qlearn.py) – Classical and perseveration-augmented Q-learning with differential evolution fitting.
- [banditpy/models/regression_models.py](banditpy/models/regression_models.py) – Logistic regression over choice/reward histories with coefficient summaries.
- [banditpy/models/thompson_models.py](banditpy/models/thompson_models.py) – Discounted Thompson sampling with flexible learning-rate tying and posterior inspection.
- [banditpy/models/ucb.py](banditpy/models/ucb.py) – Classic, Bayesian, and reinforcement-learning-flavored UCB families with softmax decision policies.
- [banditpy/analyses](banditpy/analyses) – Trial-wise and session-wise behavioral metrics, exemplified by switching probability.

## Data Expectations
- Two-armed tasks assume choices encoded as 1/2 with Bernoulli rewards (0/1).
- `.dat` logs must provide event codes, arguments, outcome probabilities, and timestamps as produced by the DIBA lab acquisition software.
- For direct construction of Bandit2Arm objects, provide arrays for `probs`, `choices`, `rewards`, and `session_ids`, optionally adding block/window labels and absolute timestamps.

## Contributing
- File issues or pull requests describing the dataset or model you want to support.
- Follow PEP8, document non-obvious logic inline, and add minimal usage examples for new components.
- Run model fits on a small synthetic dataset to ensure likelihood functions remain well-behaved before submitting changes.
