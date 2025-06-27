from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from ..models import BanditTrainer2Arm


def plot_rnn_activity_dynamics(
    model: BanditTrainer2Arm, reward_probs, n_trials=100, same_fig=True, cmap="hot"
):
    reward_probs = np.array(reward_probs)
    n_combs = reward_probs.shape[0]

    hidden_states, choices = [], []
    for r, rw in enumerate(reward_probs):
        data_mdl = model.analyze_hidden_states(reward_probs=rw[m][0], n_trials=n_trials)
        hidden_states.append(np.array(data_mdl["hidden_states"]))
        choices.append(np.array(data_mdl["actions"]))

    hidden_states = np.concatenate(hidden_states, axis=0)
    choices = np.concatenate(choices, axis=0)

    # ------ Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    hidden_states_2d = pca.fit_transform(hidden_states)

    if same_fig:
        fig, ax = plt.subplots(1, 1)
        axs = n_combs * [ax]
    else:
        fig, axs = plt.subplots(1, n_combs)

    for i in enumerate(n_combs):

        ax = axs[i]
        ax.scatter(
            hidden_states_2d[i * n_trials : (i + 1) * n_trials, 0],
            hidden_states_2d[i * n_trials : (i + 1) * n_trials, 1],
            c=range(n_trials),
            cmap=cmap,
            s=5,
        )

        ax.plot(
            hidden_states_2d[i * n_trials : (i + 1) * n_trials, 0],
            hidden_states_2d[i * n_trials : (i + 1) * n_trials, 1],
            "k-",
            alpha=0.3,
            linewidth=1,
        )

    fig.suptitle("RNN Activity Dynamics")
    fig.supxlabel("PCA Component 1")
    fig.supylabel("PCA Component 2")
