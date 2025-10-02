import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os
import math
from tqdm import tqdm


class BanditLSTMModel(nn.Module):
    """
    A recurrent neural network model inspired by the architecture described in
    meta-reinforcement learning papers (e.g., Wang et al., 2018).
    Suitable for tasks like the N-armed bandit problem.

    The input x_t at time t typically consists of:
    - Previous action a_{t-1} (one-hot encoded).
    - Previous reward r_{t-1} (scalar).
    - Optionally, a current observation obs_t (scalar or vector).
    For a 2-armed bandit with no separate observation, input_size would be 2 (action) + 1 (reward) = 3.
    """

    def __init__(self, input_size, hidden_size, num_actions, recurrent_type="lstm"):
        """
        Args:
            input_size (int): Dimensionality of the input vector x_t.
            hidden_size (int): Number of units in the recurrent layer (d_h in the paper).
            num_actions (int): Number of possible actions (for the policy output).
            recurrent_type (str): Type of recurrent layer, 'lstm' or 'gru'.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.recurrent_type = recurrent_type.lower()

        if self.recurrent_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif self.recurrent_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("Invalid recurrent_type. Choose 'lstm' or 'gru'.")

        # Output head for policy logits (can be Q-values for discrete actions)
        self.policy_head = nn.Linear(hidden_size, num_actions)
        # Output head for state value
        self.value_head = nn.Linear(hidden_size, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes weights according to common practices and paper descriptions.
        - RNN weights: Xavier uniform.
        - RNN biases: Zero.
        - Output layer weights: Scaled normal distribution.
        - Output layer biases: Zero.
        """
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if (
                "weight_ih" in name or "weight_hh" in name
            ):  # Input-to-hidden and hidden-to-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:  # All biases (ih and hh)
                nn.init.constant_(param.data, 0)

        # Initialize output layer weights and biases
        # Policy head
        nn.init.normal_(
            self.policy_head.weight.data,
            mean=0.0,
            std=1.0 / math.sqrt(self.hidden_size),
        )
        nn.init.constant_(self.policy_head.bias.data, 0)

        # Value head
        nn.init.normal_(
            self.value_head.weight.data, mean=0.0, std=1.0 / math.sqrt(self.hidden_size)
        )
        nn.init.constant_(self.value_head.bias.data, 0)

    def forward(self, x, hidden_state=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            hidden_state (tuple or torch.Tensor, optional): Initial hidden state for the RNN.
                                                           Defaults to None (zero-initialized by PyTorch).

        Returns:
            policy_logits (torch.Tensor): Logits for the policy distribution (or Q-values).
                                          Shape: (batch_size, sequence_length, num_actions).
            value (torch.Tensor): Estimated state value.
                                  Shape: (batch_size, sequence_length).
            new_hidden_state (tuple or torch.Tensor): The final hidden state of the RNN.
        """
        # RNN forward pass
        # rnn_out shape: (batch_size, sequence_length, hidden_size)
        # new_hidden_state is a tuple (h_n, c_n) for LSTM, or a tensor h_n for GRU
        rnn_out, new_hidden_state = self.rnn(x, hidden_state)

        # Calculate policy logits from all RNN hidden states in the sequence
        policy_logits = self.policy_head(
            rnn_out
        )  # Shape: (batch_size, sequence_length, num_actions)

        # Calculate state value from all RNN hidden states in the sequence
        value_raw = self.value_head(rnn_out)  # Shape: (batch_size, sequence_length, 1)
        value = value_raw.squeeze(-1)  # Shape: (batch_size, sequence_length)

        return policy_logits, value, new_hidden_state


class BanditTrainer2Arm:
    def __init__(
        self,
        # Model and training hparams
        input_size=3,  # 2 for one-hot action (1,2) + 1 for reward
        hidden_size=48,
        num_model_actions=2,  # Model outputs Q-values for 2 actions (0, 1)
        lr=0.00004,
        gamma=0.9,
        beta_entropy=0.045,
        beta_value=0.025,
        model_path="two_arm_task_model.pt",
        device=None,
    ):

        self.lr = lr
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.beta_value = beta_value
        self.model_path = model_path
        self.input_size = input_size
        self.num_model_actions = (
            num_model_actions  # Number of outputs from the policy head
        )
        self.train_type = None

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = BanditLSTMModel(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_actions=self.num_model_actions,  # policy_head outputs this many logits
            recurrent_type="lstm",
        ).to(self.device)

        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=self.lr, alpha=0.99
        )
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_bonus_history = []
        self.training_loss_history = []

    def _get_reward_probs(self, mode, N, low=0, high=1, decimals=1):
        """
        Generates reward probabilities for the two arms for a session.
        """
        if isinstance(mode, np.ndarray):
            if mode.shape == (2,):
                p_arm1 = np.ones(N) * mode[0]
                p_arm2 = np.ones(N) * mode[1]
            elif mode.shape == (N, 2):
                p_arm1, p_arm2 = mode[:, 0], mode[:, 1]

            self.train_type = "CustomProbabilities"

        elif isinstance(mode, str):
            match mode:
                case "Structured" | "Struc" | "S":
                    p_arm1 = np.round(
                        np.random.uniform(low, high, size=N), decimals=decimals
                    )
                    p_arm2 = np.round(1.0 - p_arm1, decimals=decimals)

                    self.train_type = "Structured"

                case "Unstructured" | "Unstruc" | "U":
                    p_arm1 = np.round(
                        np.random.uniform(low, high, size=N), decimals=decimals
                    )
                    p_arm2 = np.round(
                        np.random.uniform(low, high, size=N), decimals=decimals
                    )
                    self.train_type = "Unstructured"

        elif isinstance(mode, list):
            assert (
                len(mode) == 2
            ), "Reward probabilities list must have exactly 2 elements."
            p_arm1 = mode[0] * np.ones(N)
            p_arm2 = mode[1] * np.ones(N)
            self.train_type = "CustomProbabilities"

        else:
            raise ValueError(
                "Invalid mode. Use 'Structured'/'Struc'/'S', 'Unstructured'/'Unstruc'/'U', or a list of probabilities of length 2, or a numpy array of shape (2,) or (N, 2)."
            )

        # Ensure probabilities are valid
        if ~(np.all(p_arm1 <= 1) and np.all(p_arm2 <= 1)):
            raise ValueError("Reward probabilities must be between 0 and 1.")

        return np.array([p_arm1, p_arm2]).T  # Index 0 for arm 1, index 1 for arm 2

    def _generate_input(self, env_action, reward):
        """
        Generates the input tensor for the model.
        Args:
            env_action (int): The action taken in the environment (1 or 2).
            reward (float): The reward received.
        Returns:
            torch.Tensor: Input tensor for the model.
        """
        # Model expects 0-indexed one-hot for action part of input
        # input_vec: [action1_one_hot, action2_one_hot, reward]
        input_vec = torch.zeros(
            self.input_size, device=self.device
        )  # self.input_size should be 3

        # env_action is 1 or 2. num_model_actions is 2.
        # Action 1 maps to index 0, Action 2 maps to index 1.
        input_vec[env_action - 1] = 1
        input_vec[self.num_model_actions] = (
            reward  # Reward at the last position (index 2)
        )
        return input_vec

    def _discounted_return(self, rewards):
        G = []
        R_val = 0.0
        for r in reversed(rewards):
            R_val = r + self.gamma * R_val
            G.insert(0, R_val)
        return torch.tensor(G, dtype=torch.float32, device=self.device)

    def _reset_idxs(self, n_sessions):
        """
        Generates indices at which to reset the LSTM hidden state.
        Mimics animal training where animals may do 1, 2, or 3 sessions before a break.
        """
        reset_freq = np.array([1, 2, 3])  # Every 1, 2, or 3 sessions
        reset_idxs = np.cumsum(
            np.random.choice(reset_freq, size=n_sessions // reset_freq.min())
        )
        reset_idxs = reset_idxs[reset_idxs < n_sessions]
        # Always reset at the start of the first session
        reset_idxs = [0] + reset_idxs.tolist()
        return reset_idxs

    def train(
        self,
        mode,
        n_sessions=10000,
        n_trials=200,
        return_df=False,
        save_model=False,
        progress_bar=True,
        clip_norm=1.0,
        **prob_kwargs,
    ):
        print(f"Starting training for {n_sessions} {self.train_type} sessions...")
        reward_probs = self._get_reward_probs(mode, N=n_sessions, **prob_kwargs)

        training_data = []
        reset_idxs = self._reset_idxs(n_sessions)

        for session_idx in tqdm(range(n_sessions), disable=not progress_bar):
            session_reward_probs = reward_probs[session_idx]

            input_tensors_for_update, model_actions_taken, rewards_received = [], [], []

            current_input_for_model = torch.zeros(
                (1, 1, self.input_size), device=self.device
            )

            if session_idx in reset_idxs:
                lstm_hidden_state = None  # reset hidden state

            for _ in range(n_trials):
                policy_logits_step, _, lstm_hidden_state = self.model(
                    current_input_for_model, lstm_hidden_state
                )

                prob_step = F.softmax(policy_logits_step.squeeze(0), dim=-1)
                dist_step = torch.distributions.Categorical(prob_step)

                # Greedy action is not preferable, limits exploration
                # model_action = torch.argmax(prob_step).item()  # Greedy action (0 or 1)
                model_action = dist_step.sample().item()

                env_action = model_action + 1  # Convert to 1 or 2

                # session_reward_probs is [p_for_arm1, p_for_arm2]. model_action 0 corresponds to arm1.
                reward = (
                    1.0 if random.random() < session_reward_probs[model_action] else 0.0
                )

                model_actions_taken.append(model_action)
                rewards_received.append(reward)

                next_input_tensor = self._generate_input(env_action, reward)
                input_tensors_for_update.append(next_input_tensor)
                current_input_for_model = next_input_tensor.unsqueeze(0).unsqueeze(0)

                training_data.append(
                    {
                        "session_id": session_idx + 1,
                        "chosen_action": env_action,
                        "reward": reward,
                        "arm1_reward_prob": session_reward_probs[0],
                        "arm2_reward_prob": session_reward_probs[1],
                    }
                )

            G = self._discounted_return(rewards_received)
            x_seq_tensor = torch.stack(input_tensors_for_update).unsqueeze(0)

            policy_logits_seq, value_estimates_seq, _ = self.model(x_seq_tensor)
            policy_logits_seq = policy_logits_seq.squeeze(0)
            value_estimates_seq = value_estimates_seq.squeeze(0)

            actions_tensor = torch.tensor(
                model_actions_taken, dtype=torch.long, device=self.device
            )
            log_probs = torch.distributions.Categorical(
                logits=policy_logits_seq
            ).log_prob(actions_tensor)
            advantage = G - value_estimates_seq
            policy_loss = -(log_probs * advantage.detach()).mean()
            value_loss = self.beta_value * advantage.pow(2).mean()
            dist_entropy = (
                torch.distributions.Categorical(logits=policy_logits_seq)
                .entropy()
                .mean()
            )
            entropy_bonus = -self.beta_entropy * dist_entropy
            loss = policy_loss + value_loss + entropy_bonus

            self.policy_loss_history.append(policy_loss.item())
            self.value_loss_history.append(value_loss.item())
            self.entropy_bonus_history.append(entropy_bonus.item())
            self.training_loss_history.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
            self.optimizer.step()

        final_avg_loss = (
            np.mean(self.training_loss_history[-100:])
            if self.training_loss_history
            else float("nan")
        )
        print(f"Training complete. Final avg loss: {final_avg_loss:.4f}")
        self.model._is_trained = True  # 👈 mark as trained

        if save_model:
            self.save_model()

        if return_df:
            print("Returning training results as DataFrame.")
            df_training_results = pd.DataFrame(training_data)
            return df_training_results
        else:
            # print("Training results not returned as DataFrame.")
            return None

    def evaluate(
        self, mode, n_sessions=200, n_trials=200, progress_bar=True, **prob_kwargs
    ):
        print("Starting evaluation with fixed weights...")
        # try:
        #     self.load_model()  # Loads model and sets to eval mode
        # except FileNotFoundError:
        #     print(f"Evaluation failed: Model file not found at {self.model_path}.")
        #     return pd.DataFrame()

        if not hasattr(self.model, "_is_trained") or not self.model._is_trained:
            try:
                self.load_model()
            except FileNotFoundError:
                print(f"Evaluation failed: Model file not found at {self.model_path}.")
                return pd.DataFrame()

        reward_probs = self._get_reward_probs(mode, N=n_sessions, **prob_kwargs)
        evaluation_data = []
        reset_idxs = self._reset_idxs(n_sessions)

        session_group = 0
        for session_idx in tqdm(range(n_sessions), disable=not progress_bar):
            session_reward_probs = reward_probs[session_idx]

            current_input_for_model = torch.zeros(
                (1, 1, self.input_size), device=self.device
            )

            if session_idx in reset_idxs:
                lstm_hidden_state = None
                session_group += 1  # Increment session_group for each reset

            for _ in range(n_trials):
                with torch.no_grad():
                    policy_logits_step, _, lstm_hidden_state = self.model(
                        current_input_for_model, lstm_hidden_state
                    )

                prob_step = F.softmax(policy_logits_step.squeeze(0), dim=-1)
                # Keeping the model action deterministic for evaluation as we want to see if the model learned the optimal policy.
                model_action = torch.argmax(prob_step).item()  # Greedy action (0 or 1)
                # dist_step = torch.distributions.Categorical(prob_step)
                # model_action = dist_step.sample().item()

                env_action = model_action + 1  # Convert to 1 or 2

                reward = (
                    1.0 if random.random() < session_reward_probs[model_action] else 0.0
                )

                evaluation_data.append(
                    {
                        "session_id": session_idx + 1,
                        "session_group": session_group,
                        "chosen_action": env_action,
                        "reward": reward,
                        "arm1_reward_prob": session_reward_probs[0],
                        "arm2_reward_prob": session_reward_probs[1],
                    }
                )

                next_input_tensor = self._generate_input(env_action, reward)
                current_input_for_model = next_input_tensor.unsqueeze(0).unsqueeze(0)

        df_evaluation_results = pd.DataFrame(evaluation_data)
        print("Evaluation complete.")
        return df_evaluation_results

    def save_model(self):
        """
        Saves both the model state dict and training loss history.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "policy_loss_history": self.policy_loss_history,
            "value_loss_history": self.value_loss_history,
            "entropy_bonus_history": self.entropy_bonus_history,
            "training_loss_history": self.training_loss_history,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr": self.lr,
            "input_size": self.input_size,
            "hidden_size": self.model.hidden_size,
            "beta_entropy": self.beta_entropy,
            "beta_value": self.beta_value,
            "gamma": self.gamma,
        }
        torch.save(checkpoint, self.model_path)
        print(f"Model and training history saved to {self.model_path}")

    @staticmethod
    def load_model(model_path, device="cpu", verbose=True):
        # device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found at {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        input_size = checkpoint["input_size"]
        hidden_size = checkpoint["hidden_size"]
        beta_entropy = checkpoint.get("beta_entropy", 0.045)
        beta_value = checkpoint.get("beta_value", 0.025)
        gamma = checkpoint.get("gamma", 0.9)
        lr = checkpoint.get("lr", 0.00004)

        # Create trainer instance with matching config
        trainer = BanditTrainer2Arm(
            input_size=input_size,
            hidden_size=hidden_size,
            beta_entropy=beta_entropy,
            beta_value=beta_value,
            gamma=gamma,
            lr=lr,
            model_path=model_path,
            device=device,
        )

        # Load model weights and optimizer
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()
        if "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.policy_loss_history = checkpoint.get("policy_loss_history", [])
        trainer.value_loss_history = checkpoint.get("value_loss_history", [])
        trainer.entropy_bonus_history = checkpoint.get("entropy_bonus_history", [])
        trainer.training_loss_history = checkpoint.get("training_loss_history", [])

        trainer.model._is_trained = True
        if verbose:
            print(f"Restored BanditTrainer2Arm from checkpoint: {model_path}")
        return trainer

    def get_model_weights(self):
        """
        Returns the model weights as a dictionary.
        """
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def analyze_weights(self):
        """
        Analyze the learned weights of the network.
        """
        weights = self.get_model_weights()

        analysis = {}

        # RNN weights analysis
        rnn_weights = {k: v for k, v in weights.items() if "rnn" in k}

        analysis["rnn_weight_stats"] = {}
        for name, weight in rnn_weights.items():
            analysis["rnn_weight_stats"][name] = {
                "mean": float(np.mean(weight)),
                "std": float(np.std(weight)),
                "min": float(np.min(weight)),
                "max": float(np.max(weight)),
            }

        # Policy head analysis
        policy_weights = weights["policy_head.weight"]
        analysis["policy_head"] = {
            "weights": policy_weights,
            "bias": weights["policy_head.bias"],
            "action_preferences": weights[
                "policy_head.bias"
            ],  # Bias shows initial action preference
        }

        return analysis

    def plot_weight_analysis(self):
        """
        Visualize weight distributions and patterns.
        """
        import matplotlib.pyplot as plt

        weights = self.get_model_weights()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. RNN weight distributions
        rnn_ih = weights["rnn.weight_ih_l0"].flatten()
        rnn_hh = weights["rnn.weight_hh_l0"].flatten()

        axes[0, 0].hist(rnn_ih, alpha=0.7, label="Input-to-Hidden", bins=30)
        axes[0, 0].hist(rnn_hh, alpha=0.7, label="Hidden-to-Hidden", bins=30)
        axes[0, 0].set_title("RNN Weight Distributions")
        axes[0, 0].legend()

        # 2. Policy head weights heatmap
        policy_weights = weights["policy_head.weight"]
        im = axes[0, 1].imshow(policy_weights, aspect="auto", cmap="RdBu")
        axes[0, 1].set_title("Policy Head Weights")
        axes[0, 1].set_xlabel("Hidden Units")
        axes[0, 1].set_ylabel("Actions")
        plt.colorbar(im, ax=axes[0, 1])

        # 3. Value head weights
        value_weights = weights["value_head.weight"].flatten()
        axes[1, 0].bar(range(len(value_weights)), value_weights)
        axes[1, 0].set_title("Value Head Weights")
        axes[1, 0].set_xlabel("Hidden Units")
        axes[1, 0].set_ylabel("Weight Value")

        # 4. Bias comparison
        policy_bias = weights["policy_head.bias"]
        value_bias = weights["value_head.bias"]

        axes[1, 1].bar(["Action 1", "Action 2"], policy_bias, label="Policy Bias")
        axes[1, 1].bar(["Value"], value_bias, label="Value Bias", alpha=0.7)
        axes[1, 1].set_title("Output Layer Biases")
        axes[1, 1].legend()

        return fig

    def analyze_hidden_states(self, reward_probs=[0.3, 0.7], n_trials=100):
        """
        Analyze hidden state evolution during a single session.
        Returns hidden states, actions, and rewards for analysis.

        Args:
            reward_probs (list): Fixed reward probabilities for the two arms
            n_trials (int): Number of trials to analyze

        Returns:
            dict: Dictionary containing hidden states, actions, rewards, etc.
        """
        self.model.eval()

        session_data = {
            "hidden_states": [],
            "cell_states": [],  # For LSTM
            "actions": [],
            "rewards": [],
            "policy_logits": [],
            "value_estimates": [],
        }

        current_input = torch.zeros((1, 1, self.input_size), device=self.device)
        lstm_hidden_state = None

        for trial in range(n_trials):
            with torch.no_grad():
                policy_logits, value_est, lstm_hidden_state = self.model(
                    current_input, lstm_hidden_state
                )

                # Store hidden and cell states
                h_n, c_n = lstm_hidden_state
                session_data["hidden_states"].append(h_n.squeeze().cpu().numpy())
                session_data["cell_states"].append(c_n.squeeze().cpu().numpy())
                session_data["policy_logits"].append(
                    policy_logits.squeeze().cpu().numpy()
                )
                session_data["value_estimates"].append(
                    value_est.squeeze().cpu().numpy()
                )

                # Sample action (use sampling, not greedy, for better analysis)
                prob_step = F.softmax(policy_logits.squeeze(0), dim=-1)
                dist_step = torch.distributions.Categorical(prob_step)
                model_action = dist_step.sample().item()
                env_action = model_action + 1

                # Generate reward
                reward = 1.0 if random.random() < reward_probs[model_action] else 0.0

                session_data["actions"].append(env_action)
                session_data["rewards"].append(reward)

                # Prepare next input
                next_input = self._generate_input(env_action, reward)
                current_input = next_input.unsqueeze(0).unsqueeze(0)

        return session_data

    def plot_rnn_activation_evolution(
        self, reward_probs=[0.3, 0.7], n_trials=50, session_idx=0
    ):
        """
        Plot the evolution of RNN activation patterns during individual trials.

        Args:
            reward_probs: Fixed reward probabilities for the two arms
            n_trials: Number of trials to analyze
            session_idx: Which session to analyze (for title)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        self.model.eval()

        # Storage for activation data
        hidden_states = []
        cell_states = []
        actions = []
        rewards = []
        policy_probs = []
        value_estimates = []

        # Initialize
        current_input = torch.zeros((1, 1, self.input_size), device=self.device)
        lstm_hidden_state = None

        # Collect data for each trial
        for trial in range(n_trials):
            with torch.no_grad():
                policy_logits, value_est, lstm_hidden_state = self.model(
                    current_input, lstm_hidden_state
                )

                # Store activations
                h_n, c_n = lstm_hidden_state
                hidden_states.append(h_n.squeeze().cpu().numpy())
                cell_states.append(c_n.squeeze().cpu().numpy())

                # Store policy and value
                probs = F.softmax(policy_logits.squeeze(), dim=-1)
                policy_probs.append(probs.cpu().numpy())
                value_estimates.append(value_est.squeeze().cpu().numpy())

                # Sample action
                model_action = torch.distributions.Categorical(probs).sample().item()
                env_action = model_action + 1
                actions.append(env_action)

                # Generate reward
                reward = 1.0 if np.random.random() < reward_probs[model_action] else 0.0
                rewards.append(reward)

                # Prepare next input
                next_input = self._generate_input(env_action, reward)
                current_input = next_input.unsqueeze(0).unsqueeze(0)

        # Convert to arrays
        hidden_states = np.array(hidden_states)  # Shape: (n_trials, hidden_size)
        cell_states = np.array(cell_states)
        policy_probs = np.array(policy_probs)
        value_estimates = np.array(value_estimates)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Create comprehensive plot
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # 1. Hidden state heatmap
        im1 = axes[0, 0].imshow(
            hidden_states.T, aspect="auto", cmap="RdBu", interpolation="nearest"
        )
        axes[0, 0].set_title("Hidden State Evolution")
        axes[0, 0].set_xlabel("Trial")
        axes[0, 0].set_ylabel("Hidden Unit")
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. Cell state heatmap
        im2 = axes[0, 1].imshow(
            cell_states.T, aspect="auto", cmap="RdBu", interpolation="nearest"
        )
        axes[0, 1].set_title("Cell State Evolution")
        axes[0, 1].set_xlabel("Trial")
        axes[0, 1].set_ylabel("Hidden Unit")
        plt.colorbar(im2, ax=axes[0, 1])

        # 3. Policy probabilities and actions
        axes[1, 0].plot(policy_probs[:, 0], label="Prob(Action 1)", linewidth=2)
        axes[1, 0].plot(policy_probs[:, 1], label="Prob(Action 2)", linewidth=2)

        # Color background based on actual actions taken
        for trial in range(n_trials):
            color = "lightblue" if actions[trial] == 1 else "lightcoral"
            axes[1, 0].axvspan(trial - 0.5, trial + 0.5, alpha=0.3, color=color)

        axes[1, 0].set_title("Policy Probabilities & Actions")
        axes[1, 0].set_xlabel("Trial")
        axes[1, 0].set_ylabel("Probability")
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)

        # 4. Value estimates and rewards
        axes[1, 1].plot(
            value_estimates, label="Value Estimate", linewidth=2, color="green"
        )

        # Show rewards as scatter
        reward_trials = np.where(rewards == 1)[0]
        no_reward_trials = np.where(rewards == 0)[0]
        axes[1, 1].scatter(
            reward_trials,
            np.ones(len(reward_trials)),
            color="gold",
            s=50,
            label="Reward",
            zorder=5,
        )
        axes[1, 1].scatter(
            no_reward_trials,
            np.zeros(len(no_reward_trials)),
            color="red",
            s=50,
            label="No Reward",
            zorder=5,
        )

        axes[1, 1].set_title("Value Estimates & Rewards")
        axes[1, 1].set_xlabel("Trial")
        axes[1, 1].set_ylabel("Value / Reward")
        axes[1, 1].legend()

        # 5. Hidden state PCA evolution
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        hidden_pca = pca.fit_transform(hidden_states)

        # Color by trial number
        scatter = axes[2, 0].scatter(
            hidden_pca[:, 0], hidden_pca[:, 1], c=range(n_trials), cmap="viridis", s=60
        )
        axes[2, 0].plot(hidden_pca[:, 0], hidden_pca[:, 1], "k-", alpha=0.3)
        axes[2, 0].set_title(
            f"Hidden State Trajectory (PCA)\nVar explained: {pca.explained_variance_ratio_.sum():.2f}"
        )
        axes[2, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f})")
        axes[2, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f})")
        plt.colorbar(scatter, ax=axes[2, 0], label="Trial")

        # 6. Action-reward contingency
        action_reward_data = []
        for trial in range(n_trials):
            action_reward_data.append([trial, actions[trial], rewards[trial]])

        action_reward_df = pd.DataFrame(
            action_reward_data, columns=["Trial", "Action", "Reward"]
        )

        # Plot action choices over time with reward outcomes
        for action in [1, 2]:
            action_trials = action_reward_df[action_reward_df["Action"] == action]
            rewarded = action_trials[action_trials["Reward"] == 1]
            unrewarded = action_trials[action_trials["Reward"] == 0]

            axes[2, 1].scatter(
                rewarded["Trial"],
                rewarded["Action"],
                color="green",
                s=60,
                alpha=0.8,
                label=f"Action {action} (Rewarded)" if action == 1 else None,
            )
            axes[2, 1].scatter(
                unrewarded["Trial"],
                unrewarded["Action"],
                color="red",
                s=60,
                alpha=0.8,
                label=f"Action {action} (No Reward)" if action == 1 else None,
            )

        axes[2, 1].set_title(f"Action-Reward History\nArm Probs: {reward_probs}")
        axes[2, 1].set_xlabel("Trial")
        axes[2, 1].set_ylabel("Action")
        axes[2, 1].set_yticks([1, 2])
        axes[2, 1].legend()

        plt.suptitle(f"RNN Activation Evolution - Session {session_idx}", fontsize=16)

        return fig, {
            "hidden_states": hidden_states,
            "cell_states": cell_states,
            "policy_probs": policy_probs,
            "value_estimates": value_estimates,
            "actions": actions,
            "rewards": rewards,
            "hidden_pca": hidden_pca,
        }

    def _calculate_entropy(self, actions):
        """Calculate entropy of action distribution"""
        if len(actions) == 0:
            return 0

        # Calculate probabilities for each action
        p1 = (actions == 1).mean()
        p2 = (actions == 2).mean()

        # Calculate entropy (max entropy = 1 for 2 actions)
        entropy = 0
        if p1 > 0:
            entropy -= p1 * np.log2(p1)
        if p2 > 0:
            entropy -= p2 * np.log2(p2)

        return entropy


class GRUCellEq2(nn.Module):
    """
    Custom GRU cell implementing the equations in the text (Eq. 2 style):
      r_t = sigma(W^r_x x_t + b^r_x + W^r_h h_{t-1} + b^r_h)
      z_t = sigma(W^z_x x_t + b^z_x + W^z_h h_{t-1} + b^z_h)
      n_t = tanh(W^n_x x_t + b^n_x + r_t ∘ (W^n_h h_{t-1} + b^n_h))
      h_t = (1 - z_t) ∘ n_t + z_t ∘ h_{t-1}
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H, D = hidden_size, input_size
        # input-to-hidden
        self.Wxr = nn.Linear(D, H, bias=True)  # bias == b^r_x
        self.Wxz = nn.Linear(D, H, bias=True)  # bias == b^z_x
        self.Wxn = nn.Linear(D, H, bias=True)  # bias == b^n_x
        # hidden-to-hidden (no bias here; we add explicit b^·_h below)
        self.Whr = nn.Linear(H, H, bias=False)
        self.Whz = nn.Linear(H, H, bias=False)
        self.Whn = nn.Linear(H, H, bias=False)
        # hidden biases
        self.brh = nn.Parameter(torch.zeros(H))  # b^r_h
        self.bzh = nn.Parameter(torch.zeros(H))  # b^z_h
        self.bnh = nn.Parameter(torch.zeros(H))  # b^n_h
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.Wxr, self.Wxz, self.Wxn, self.Whr, self.Whz, self.Whn]:
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.brh, 0.0)
        nn.init.constant_(self.bzh, 0.0)
        nn.init.constant_(self.bnh, 0.0)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        r_t = torch.sigmoid(self.Wxr(x_t) + self.Whr(h_prev) + self.brh)
        z_t = torch.sigmoid(self.Wxz(x_t) + self.Whz(h_prev) + self.bzh)
        n_t = torch.tanh(self.Wxn(x_t) + r_t * (self.Whn(h_prev) + self.bnh))
        h_t = (1.0 - z_t) * n_t + z_t * h_prev
        return h_t


class SwitchGRUCell(nn.Module):
    """
    Switching-GRU (sGRU) cell for discrete inputs (as described in the text).
    The effective parameters are a convex combination of K expert parameter sets
    selected by a one-hot (or soft) selector u_t ∈ R^K computed from x_t.

    For each gate g ∈ {r,z,n} we keep K expert weights:
      W^g_x[k], W^g_h[k], b^g_x[k], b^g_h[k],  k = 1..K
    Effective parameters at t are:
      W^g_x(t) = Σ_k u_t[k] W^g_x[k]  (and similarly for others)
    Then apply the GRU equations (Eq. 2) with those effective parameters.
    """

    def __init__(self, input_size: int, hidden_size: int, num_modes: int):
        super().__init__()
        self.K = num_modes
        H, D, K = hidden_size, input_size, num_modes

        # Expert banks
        self.Wxr = nn.Parameter(torch.empty(K, H, D))
        self.Wxz = nn.Parameter(torch.empty(K, H, D))
        self.Wxn = nn.Parameter(torch.empty(K, H, D))

        self.Whr = nn.Parameter(torch.empty(K, H, H))
        self.Whz = nn.Parameter(torch.empty(K, H, H))
        self.Whn = nn.Parameter(torch.empty(K, H, H))

        self.bxr = nn.Parameter(torch.zeros(K, H))
        self.bxz = nn.Parameter(torch.zeros(K, H))
        self.bxn = nn.Parameter(torch.zeros(K, H))

        self.brh = nn.Parameter(torch.zeros(K, H))
        self.bzh = nn.Parameter(torch.zeros(K, H))
        self.bnh = nn.Parameter(torch.zeros(K, H))

        self.reset_parameters()

    def reset_parameters(self):
        def xavier_bank(W):
            for k in range(W.shape[0]):
                nn.init.xavier_uniform_(W[k])

        for bank in [self.Wxr, self.Wxz, self.Wxn, self.Whr, self.Whz, self.Whn]:
            xavier_bank(bank)
        for b in [self.bxr, self.bxz, self.bxn, self.brh, self.bzh, self.bnh]:
            nn.init.constant_(b, 0.0)

    def _mix(self, bank: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        # bank: (K, H, D) or (K, H, H) or (K, H); u_t: (B, K)
        # returns mixed params with leading batch dim
        if bank.dim() == 3:
            # (B, H, D/H)
            return torch.einsum("khd,bk->bhd", bank, u_t)
        elif bank.dim() == 2:
            return torch.einsum("kh,bk->bh", bank, u_t)
        else:
            raise ValueError("Unexpected bank dim")

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, u_t: torch.Tensor
    ) -> torch.Tensor:
        # x_t: (B, D), h_prev: (B, H), u_t: (B, K)
        Wxr = self._mix(self.Wxr, u_t)  # (B,H,D)
        Wxz = self._mix(self.Wxz, u_t)
        Wxn = self._mix(self.Wxn, u_t)

        Whr = self._mix(self.Whr, u_t)  # (B,H,H)
        Whz = self._mix(self.Whz, u_t)
        Whn = self._mix(self.Whn, u_t)

        bxr = self._mix(self.bxr, u_t)  # (B,H)
        bxz = self._mix(self.bxz, u_t)
        bxn = self._mix(self.bxn, u_t)

        brh = self._mix(self.brh, u_t)  # (B,H)
        bzh = self._mix(self.bzh, u_t)
        bnh = self._mix(self.bnh, u_t)

        # affine ops with batched weights
        def aff_x(W, x, b):  # (B,H,D) @ (B,D) + (B,H)
            return torch.einsum("bhd,bd->bh", W, x) + b

        def aff_h(W, h, b):
            return torch.einsum("bhh,bh->bh", W, h) + b

        r_t = torch.sigmoid(aff_x(Wxr, x_t, bxr) + aff_h(Whr, h_prev, brh))
        z_t = torch.sigmoid(aff_x(Wxz, x_t, bxz) + aff_h(Whz, h_prev, bzh))
        n_t = torch.tanh(aff_x(Wxn, x_t, bxn) + r_t * (aff_h(Whn, h_prev, bnh)))
        h_t = (1.0 - z_t) * n_t + z_t * h_prev
        return h_t


class BanditSwitchRNN(nn.Module):
    """
    RNN policy model with either:
      - Vanilla GRU cell (GRUCellEq2), or
      - Switching GRU cell (SwitchGRUCell) with K modes selected from inputs.

    Output layer:
      - Fully connected readout: s_t = Θ h_t + b  (recommended; default)
      - Optional diagonal readout when hidden_size == num_actions:
            s_i(t) = θ_i * h_i(t) + b_i
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_actions: int,
        cell_type: str = "gru",  # "gru" or "sgru"
        num_modes: int = 4,  # only used for sgru
        diagonal_readout: bool = False,  # if True, requires hidden_size == num_actions
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.cell_type = cell_type.lower()
        self.num_modes = num_modes
        self.diagonal_readout = diagonal_readout and (hidden_size == num_actions)

        if self.cell_type == "gru":
            self.cell = GRUCellEq2(input_size, hidden_size)
        elif self.cell_type == "sgru":
            self.cell = SwitchGRUCell(input_size, hidden_size, num_modes=num_modes)
        else:
            raise ValueError("cell_type must be 'gru' or 'sgru'.")

        if self.diagonal_readout:
            # θ ∈ R^{A}, b ∈ R^{A}, applied element-wise to h (A must equal H)
            self.theta = nn.Parameter(torch.zeros(num_actions))
            self.bias = nn.Parameter(torch.zeros(num_actions))
            nn.init.normal_(self.theta, mean=0.0, std=1.0 / math.sqrt(hidden_size))
            nn.init.constant_(self.bias, 0.0)
        else:
            self.readout = nn.Linear(hidden_size, num_actions)
            nn.init.normal_(
                self.readout.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size)
            )
            nn.init.constant_(self.readout.bias, 0.0)

    @staticmethod
    def default_selector(x_t: torch.Tensor, num_modes: int) -> torch.Tensor:
        """
        Build selector u_t from x_t for a 2-armed bandit with input [onehot(a_{t-1}), r_{t-1}],
        i.e., x_t = [a1,a2,r], where a1+a2 ∈ {0,1}, r ∈ {0,1}.
        Modes: 0:(a=1,r=0), 1:(a=2,r=0), 2:(a=1,r=1), 3:(a=2,r=1)  → K=4.
        If num_modes != 4, returns a uniform (no-switch) selector.
        """
        B, D = x_t.shape
        if num_modes != 4 or D < 3:
            return torch.full((B, num_modes), 1.0 / num_modes, device=x_t.device)
        a = torch.argmax(x_t[:, :2], dim=-1)  # 0 or 1
        r = (x_t[:, 2] > 0.5).long()  # 0 or 1
        idx = a + 2 * r  # 0..3
        u = F.one_hot(idx, num_classes=4).float()
        return u

    def forward(
        self, x: torch.Tensor, h0: torch.Tensor | None = None, selector_fn=None
    ):
        """
        x: (B, T, D)
        Returns:
          logits: (B, T, A)
          h: final hidden (B, H)
        """
        B, T, D = x.shape
        H, A = self.hidden_size, self.num_actions
        h = torch.zeros(B, H, device=x.device) if h0 is None else h0

        logits_list = []
        for t in range(T):
            x_t = x[:, t, :]
            if self.cell_type == "sgru":
                if selector_fn is None:
                    u_t = self.default_selector(x_t, self.num_modes)
                else:
                    u_t = selector_fn(x_t)  # expects (B,K)
                h = self.cell(x_t, h, u_t)
            else:
                h = self.cell(x_t, h)

            if self.diagonal_readout:
                # element-wise mapping then identity to actions (requires A==H)
                s_t = self.theta * h + self.bias
            else:
                s_t = self.readout(h)
            logits_list.append(s_t.unsqueeze(1))

        logits = torch.cat(logits_list, dim=1)
        return logits, h


class BanditModelFreeAgent2Arm:
    """Model-free cognitive-style agent for 2-armed bandit tasks.

    This class mirrors the functionality provided by model-free variants
    in `tinyRNN`'s `MABCogAgent` but is implemented locally (no numba, no
    external tinyRNN dependency) and focused on two-armed tasks.

    Supported variants (string argument `variant`):
      - 'mfd'   : Model-free with optional global decay (Q <- beta * Q) then chosen update
      - 'mfdp'  : Same as 'mfd' plus perseveration bias parameter (rho)
      - 'mflb'  : Model-free learn-all (binary rewards) — simultaneous chosen & unchosen updates

    Parameterizations:
      variant='mfd' (decay=True): params = [alpha, beta_decay, inv_temp]
      variant='mfd' (decay=False): params = [alpha, inv_temp]
      variant='mfdp': as above + perseveration rho (added last)
      variant='mflb': params = [alpha_c, util_c_r0, util_c_r1, alpha_u, util_u_r0, util_u_r1, inv_temp]

    Perseveration (rho) is implemented as adding +rho to the logit of the
    previously chosen action (0-based). This can capture stay/switch bias.

    Methods:
      simulate(n_trials, reward_probs, params=None, seed=None, greedy_eval=False)
      log_likelihood(choices, rewards, params)
      fit(choices, rewards, n_starts=50, method='auto')

    Choices can be provided as 0/1 or 1/2; detection is automatic.

    Note: A lightweight optimizer is included (random multi-start + optional
    SciPy refine if available). For rigorous fitting you may still prefer
    specialized behavioral fitting pipelines.

    Example:
        agent = BanditModelFreeAgent2Arm(variant='mfd', decay=True)
        sim = agent.simulate(500, reward_probs=[0.3, 0.7])
        fit_res = agent.fit(sim['choices'], sim['rewards'])
        print(fit_res['best_params'], fit_res['neg_loglik'])
    """

    def __init__(self, variant: str = "mfd", decay: bool = True):
        self.variant = variant.lower()
        self.decay = decay if self.variant in ("mfd", "mfdp") else False
        self._set_param_spec()

    # ---------------------------- Parameter Spec ---------------------------- #
    def _set_param_spec(self):
        spec = []
        if self.variant == "mfd":
            if self.decay:
                spec = [
                    ("alpha", "unit"),
                    ("beta_decay", "unit"),
                    ("inv_temp", "pos"),
                ]
            else:
                spec = [
                    ("alpha", "unit"),
                    ("inv_temp", "pos"),
                ]
        elif self.variant == "mfdp":
            # add perseveration rho (unconstrained)
            if self.decay:
                spec = [
                    ("alpha", "unit"),
                    ("beta_decay", "unit"),
                    ("inv_temp", "pos"),
                    ("rho", "unc"),
                ]
            else:
                spec = [
                    ("alpha", "unit"),
                    ("inv_temp", "pos"),
                    ("rho", "unc"),
                ]
        elif self.variant == "mflb":
            spec = [
                ("alpha_c", "unit"),
                ("util_c_r0", "unc"),
                ("util_c_r1", "unc"),
                ("alpha_u", "unit"),
                ("util_u_r0", "unc"),
                ("util_u_r1", "unc"),
                ("inv_temp", "pos"),
            ]
        else:
            raise ValueError("variant must be one of {'mfd','mfdp','mflb'}")
        self.param_spec = spec

    # ---------------------------- Utilities ---------------------------- #
    @staticmethod
    def _softmax(logits):
        z = logits - np.max(logits)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    def _coerce_actions(self, choices):
        choices = np.asarray(choices)
        if choices.min() == 1 and choices.max() == 2:
            return choices - 1  # convert to 0/1
        return choices

    def default_params(self):
        # Provide reasonable default center points
        defaults = []
        for name, ptype in self.param_spec:
            if ptype == "unit":
                defaults.append(0.5)
            elif ptype == "pos":
                defaults.append(5.0)
            elif ptype == "unc":
                defaults.append(0.0)
        return np.array(defaults, dtype=float)

    # ---------------------------- Core Updates ---------------------------- #
    def _update_mfd(self, Q, action, reward, params):
        if self.decay:
            if self.variant == "mfdp":
                alpha, beta_decay, inv_temp, *rest = params
            else:
                alpha, beta_decay, inv_temp = params[:3]
            Q *= beta_decay
        else:
            if self.variant == "mfdp":
                alpha, inv_temp, *rest = params
            else:
                alpha, inv_temp = params[:2]
        Q[action] = (1 - alpha) * Q[action] + alpha * reward
        return Q

    def _update_mflb(self, Q, action, reward, params):
        (alpha_c, util_c_r0, util_c_r1, alpha_u, util_u_r0, util_u_r1, inv_temp) = (
            params
        )
        # Copy + unchosen decay/upweight
        if reward == 0:
            Q = alpha_u * Q + util_u_r0
            Q[action] = alpha_c * Q[action] + util_c_r0
        else:
            Q = alpha_u * Q + util_u_r1
            Q[action] = alpha_c * Q[action] + util_c_r1
        return Q

    # ---------------------------- Simulation ---------------------------- #
    def simulate(
        self, n_trials: int, reward_probs, params=None, seed=None, greedy_eval=False
    ):
        """Simulate behavior under the specified variant.

        Args:
            n_trials: number of trials
            reward_probs: list/array shape (2,) with reward probabilities for actions 0 and 1
            params: override parameter vector; if None uses defaults
            seed: random seed
            greedy_eval: if True choose argmax prob instead of sampling
        Returns dict with keys: choices (0/1), rewards (0/1), Q_history (n_trials+1,2), probs (n_trials,2)
        """
        rng = np.random.default_rng(seed)
        reward_probs = np.asarray(reward_probs, dtype=float)
        assert reward_probs.shape == (2,), "reward_probs must be length 2"
        if params is None:
            params = self.default_params()
        params = np.asarray(params, dtype=float)

        Q = np.zeros(2, dtype=float)
        Q_hist = np.zeros((n_trials + 1, 2))
        Q_hist[0] = Q
        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials, dtype=int)
        probs_arr = np.zeros((n_trials, 2))
        prev_action = None

        for t in range(n_trials):
            # Compute logits
            if self.variant in ("mfd", "mfdp"):
                if self.decay:
                    if self.variant == "mfdp":
                        alpha, beta_decay, inv_temp, *rest = params
                    else:
                        alpha, beta_decay, inv_temp = params[:3]
                else:
                    if self.variant == "mfdp":
                        alpha, inv_temp, *rest = params
                    else:
                        alpha, inv_temp = params[:2]
                logits = inv_temp * Q
                if self.variant == "mfdp":
                    rho = params[-1]
                    if prev_action is not None:
                        logits[prev_action] += rho
            elif self.variant == "mflb":
                inv_temp = params[-1]
                logits = inv_temp * Q
            else:
                raise RuntimeError

            probs = self._softmax(logits)
            probs_arr[t] = probs
            if greedy_eval:
                action = int(np.argmax(probs))
            else:
                action = int(rng.choice([0, 1], p=probs))
            reward = int(rng.random() < reward_probs[action])

            # Update
            if self.variant in ("mfd", "mfdp"):
                Q = self._update_mfd(Q, action, reward, params)
            else:
                Q = self._update_mflb(Q, action, reward, params)

            choices[t] = action
            rewards[t] = reward
            Q_hist[t + 1] = Q
            prev_action = action

        return {
            "choices": choices,
            "rewards": rewards,
            "Q_history": Q_hist,
            "probs": probs_arr,
            "params_used": params,
        }

    # ---------------------------- Log Likelihood ---------------------------- #
    def log_likelihood(self, choices, rewards, params):
        choices = self._coerce_actions(choices)
        rewards = np.asarray(rewards).astype(int)
        params = np.asarray(params, dtype=float)
        n = len(choices)
        assert len(rewards) == n
        Q = np.zeros(2, dtype=float)
        prev_action = None
        loglik = 0.0
        eps = 1e-9
        for t in range(n):
            if self.variant in ("mfd", "mfdp"):
                if self.decay:
                    if self.variant == "mfdp":
                        alpha, beta_decay, inv_temp, *rest = params
                    else:
                        alpha, beta_decay, inv_temp = params[:3]
                else:
                    if self.variant == "mfdp":
                        alpha, inv_temp, *rest = params
                    else:
                        alpha, inv_temp = params[:2]
                logits = inv_temp * Q
                if self.variant == "mfdp" and prev_action is not None:
                    rho = params[-1]
                    logits[prev_action] += rho
            elif self.variant == "mflb":
                inv_temp = params[-1]
                logits = inv_temp * Q
            else:
                raise RuntimeError
            probs = self._softmax(logits)
            action = choices[t]
            loglik += np.log(probs[action] + eps)
            reward = rewards[t]
            # Update
            if self.variant in ("mfd", "mfdp"):
                Q = self._update_mfd(Q, action, reward, params)
            else:
                Q = self._update_mflb(Q, action, reward, params)
            prev_action = action
        return loglik

    # ---------------------------- Parameter Fitting ---------------------------- #
    def _sample_random_params(self, rng):
        vals = []
        for name, ptype in self.param_spec:
            if ptype == "unit":
                vals.append(rng.uniform(0.01, 0.99))
            elif ptype == "pos":
                vals.append(np.exp(rng.uniform(np.log(0.1), np.log(20.0))))
            elif ptype == "unc":
                vals.append(rng.normal(0.0, 0.5))
        return np.array(vals)

    def fit(
        self, choices, rewards, n_starts=50, method="auto", seed=None, verbose=False
    ):
        """Fit parameters by maximizing log-likelihood.

        A light-weight routine: multi-start random search + optional SciPy refinement.
        Returns dict with best_params, neg_loglik, all_trials (list of dicts).
        """
        rng = np.random.default_rng(seed)
        choices = self._coerce_actions(choices)
        rewards = np.asarray(rewards).astype(int)

        trials_info = []
        best_ll = -np.inf
        best_params = None

        for i in range(n_starts):
            params0 = self._sample_random_params(rng)
            ll = self.log_likelihood(choices, rewards, params0)
            trials_info.append({"start_idx": i, "params": params0, "loglik": ll})
            if ll > best_ll:
                best_ll = ll
                best_params = params0

        # Optional SciPy refinement
        refined = False
        if method != "none":
            try:
                from scipy.optimize import minimize

                def neg_obj(p_raw):
                    # Project back into feasible space for unit/pos
                    p = p_raw.copy()
                    idx = 0
                    for j, (name, ptype) in enumerate(self.param_spec):
                        if ptype == "unit":
                            # squash to (0,1)
                            p[j] = 1 / (1 + np.exp(-p_raw[j]))
                        elif ptype == "pos":
                            p[j] = np.exp(p_raw[j])
                        # unc: identity
                    return -self.log_likelihood(choices, rewards, p)

                # Initialize raw params via inverse transforms
                p0_raw = []
                for val, (name, ptype) in zip(best_params, self.param_spec):
                    if ptype == "unit":
                        val = np.clip(val, 1e-6, 1 - 1e-6)
                        p0_raw.append(np.log(val) - np.log(1 - val))  # logit
                    elif ptype == "pos":
                        p0_raw.append(np.log(max(val, 1e-8)))
                    else:
                        p0_raw.append(val)
                p0_raw = np.array(p0_raw)

                res = minimize(neg_obj, p0_raw, method="L-BFGS-B")
                if res.success:
                    # Forward transform
                    p_opt = res.x.copy()
                    for j, (name, ptype) in enumerate(self.param_spec):
                        if ptype == "unit":
                            p_opt[j] = 1 / (1 + np.exp(-p_opt[j]))
                        elif ptype == "pos":
                            p_opt[j] = np.exp(p_opt[j])
                    ll_new = self.log_likelihood(choices, rewards, p_opt)
                    if ll_new > best_ll:
                        best_ll = ll_new
                        best_params = p_opt
                        refined = True
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"SciPy refinement skipped: {e}")

        return {
            "best_params": best_params,
            "neg_loglik": -best_ll,
            "loglik": best_ll,
            "variant": self.variant,
            "decay": self.decay,
            "refined": refined,
            "random_trials": trials_info,
            "param_spec": self.param_spec,
        }


class BanditModelBasedRNN(nn.Module):
    """Model-based RNN agent for two-armed bandit tasks.

    This agent explicitly learns/maintains an internal estimate of reward
    probabilities (or logits) for each arm inside a recurrent state and uses
    those to compute action logits. Unlike the purely model-free variant,
    here we encourage the hidden state to encode a belief over reward rates
    and optionally predict immediate reward probabilities directly.

    Architecture:
      - Recurrent core (GRU or LSTM) receives: [one_hot(prev_action), prev_reward]
      - Belief head: maps hidden -> reward prob logits (2 values)
      - Policy head: optionally separate mapping hidden -> action logits
        (can share weights with belief head if desired).

    Loss Components (per trial sequence):
      - Policy loss (actor): REINFORCE with (return - baseline) or simple
        advantage using a learned value head.
      - Value loss (critic): MSE between discounted returns and value estimates.
      - Belief calibration loss: BCE between predicted reward probability for
        the *action chosen* and actual reward outcome. This nudges the network
        toward explicit predictive modeling (model-based signal).

    Args:
        input_size: typically 3 (a1, a2, r_prev)
        hidden_size: recurrent hidden dimension
        num_actions: number of arms (2 for two-armed bandit)
        cell: 'gru' or 'lstm'
        belief_weight: scale of belief (reward prediction) loss
        entropy_weight: policy entropy regularization coefficient
        value_weight: scale for value loss
        shared_head: if True the policy logits are identical to reward belief logits

    Example:
        agent = BanditModelBasedRNN()
        trainer = agent.make_trainer()
        df = trainer.train_env(n_sessions=2000, n_trials=150, reward_probs_mode='Structured')
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_actions: int = 2,
        cell: str = "gru",
        belief_weight: float = 1.0,
        entropy_weight: float = 0.01,
        value_weight: float = 0.5,
        shared_head: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.cell_type = cell.lower()
        self.belief_weight = belief_weight
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.shared_head = shared_head

        if self.cell_type == "gru":
            self.core = nn.GRU(input_size, hidden_size, batch_first=True)
        elif self.cell_type == "lstm":
            self.core = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("cell must be 'gru' or 'lstm'")

        self.belief_head = nn.Linear(hidden_size, num_actions)
        if shared_head:
            self.policy_head = self.belief_head
        else:
            self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

        for m in [self.belief_head, self.value_head] + (
            [] if shared_head else [self.policy_head]
        ):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, hidden=None):
        # x: (B,T,D)
        core_out, new_hidden = self.core(x, hidden)
        belief_logits = self.belief_head(core_out)  # (B,T,A)
        policy_logits = self.policy_head(core_out)  # (B,T,A)
        values = self.value_head(core_out).squeeze(-1)  # (B,T)
        return belief_logits, policy_logits, values, new_hidden

    # ---------------- Trainer Helper ---------------- #
    def make_trainer(self, lr=3e-4, gamma=0.9, device=None):
        return _ModelBasedRNNTrainer(
            self,
            lr=lr,
            gamma=gamma,
            device=device,
            belief_weight=self.belief_weight,
            entropy_weight=self.entropy_weight,
            value_weight=self.value_weight,
        )


class _ModelBasedRNNTrainer:
    """Internal trainer for BanditModelBasedRNN.

    Provides a similar training interface to BanditTrainer2Arm but with
    additional belief prediction term.
    """

    def __init__(
        self,
        model: BanditModelBasedRNN,
        lr=3e-4,
        gamma=0.9,
        device=None,
        belief_weight=1.0,
        entropy_weight=0.01,
        value_weight=0.5,
    ):
        self.model = model
        self.gamma = gamma
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.belief_weight = belief_weight
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.train_history = []

    def _discount(self, rewards: list[float]):
        G = []
        acc = 0.0
        for r in reversed(rewards):
            acc = r + self.gamma * acc
            G.insert(0, acc)
        return torch.tensor(G, dtype=torch.float32, device=self.device)

    def _build_input(self, prev_action, prev_reward):
        x = torch.zeros(self.model.input_size, device=self.device)
        if prev_action is not None:
            x[prev_action] = 1.0
        x[self.model.num_actions] = prev_reward if prev_reward is not None else 0.0
        return x

    def _sample_reward_probs(self, mode, N, low=0, high=1, decimals=1):
        if isinstance(mode, str):


# ---------------------------------------------------------------------------
# TinyBehaviorRNN: Supervised log-likelihood behavioral RNN (paper-style)
# ---------------------------------------------------------------------------

class TinyBehaviorRNN(nn.Module):
    """Minimal GRU (or switching GRU) policy model for behavioral fitting.

    This implements the "tiny RNN" paradigm used to fit animal / human
    decision sequences via *supervised* maximum likelihood, without RL
    policy gradient. Each hidden unit corresponds to a *dynamical variable*.

    Inputs per trial (t):  x_t = [one_hot(a_{t-1}), r_{t-1}, optional state_{t-1} one-hot]
      For 2-armed bandit: dimension = 2 (action one-hot) + 1 (reward) = 3.
    Target: a_t.

    Training objective: minimize negative log-likelihood
        L = - (1/Σ mask) Σ_t mask_t log π(a_t | x_{≤t})

    Features:
      - Hidden size = d (# dynamical variables searched over)
      - Optional diagonal readout when hidden_size == num_actions
      - Optional switching GRU cell (future extension) – placeholder now
      - Learnable or fixed zero initial hidden state
      - Weight decay + early stopping

    Note: This class focuses on 2-arm (or small A) tasks; it generalizes to
    more arms if input construction supplies larger one-hot.
    """

    def __init__(
        self,
        input_size: int,
        num_actions: int,
        hidden_size: int,
        cell_type: str = 'gru',
        diagonal_readout: bool = False,
        learn_h0: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.cell_type = cell_type.lower()
        self.diagonal_readout = diagonal_readout and (hidden_size == num_actions)
        self.learn_h0 = learn_h0

        if self.cell_type != 'gru':
            raise ValueError('Currently only vanilla GRU supported for tiny model.')
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        if self.diagonal_readout:
            self.theta = nn.Parameter(torch.zeros(num_actions))
            self.bias = nn.Parameter(torch.zeros(num_actions))
            nn.init.normal_(self.theta, 0.0, 1.0 / math.sqrt(hidden_size))
        else:
            self.readout = nn.Linear(hidden_size, num_actions)
            nn.init.xavier_uniform_(self.readout.weight)
            nn.init.constant_(self.readout.bias, 0.0)

        if learn_h0:
            self.h0_param = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.register_buffer('h0_param', torch.zeros(1, 1, hidden_size))

    def forward(self, x, h0=None):
        # x: (B,T,D)
        if h0 is None:
            h0 = self.h0_param.repeat(1, x.size(0), 1)  # (1,B,H)
        out, h_n = self.rnn(x, h0)
        if self.diagonal_readout:
            logits = self.theta * out + self.bias  # broadcast (B,T,A)
        else:
            logits = self.readout(out)
        return logits, h_n

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def _prepare_sequence(session, num_actions: int, state_dim: int | None = None):
    """Convert a single session dict into (inputs, targets) torch tensors.

    session keys expected:
      'actions': array of ints (0..A-1 or 1..A)
      'rewards': array of rewards (0/1)
      optional 'states': array of ints (0..S-1)
    """
    actions = np.asarray(session['actions'])
    if actions.min() == 1:  # convert to 0-based
        actions = actions - 1
    rewards = np.asarray(session['rewards'])
    T = len(actions)
    assert len(rewards) == T
    has_states = 'states' in session and state_dim is not None
    if has_states:
        states = np.asarray(session['states'])
        assert len(states) == T

    # Build inputs at time t depend on previous action/reward
    X = []
    for t in range(T):
        if t == 0:
            a_prev_vec = np.zeros(num_actions)
            r_prev = 0.0
            if has_states:
                s_prev_vec = np.zeros(state_dim)
        else:
            a_prev_vec = np.zeros(num_actions)
            a_prev_vec[actions[t-1]] = 1.0
            r_prev = rewards[t-1]
            if has_states:
                s_prev_vec = np.zeros(state_dim)
                s_prev_vec[states[t-1]] = 1.0
        parts = [a_prev_vec, [r_prev]]
        if has_states:
            parts.append(s_prev_vec)
        X.append(np.concatenate(parts))
    X = np.stack(X, axis=0)  # (T,D)
    y = actions.copy()
    return X, y


def collate_sessions(sessions, num_actions: int, state_dim: int | None = None, device=None):
    """Pad and batch variable-length sessions.

    Returns dict with tensors: inputs (B,Tmax,D), targets (B,Tmax), mask (B,Tmax)
    """
    processed = [ _prepare_sequence(s, num_actions, state_dim) for s in sessions ]
    lengths = [x[0].shape[0] for x in processed]
    Tmax = max(lengths)
    D = processed[0][0].shape[1]
    B = len(processed)
    inputs = np.zeros((B, Tmax, D), dtype=np.float32)
    targets = np.zeros((B, Tmax), dtype=np.int64)
    mask = np.zeros((B, Tmax), dtype=np.float32)
    for i,(X,y) in enumerate(processed):
        L = X.shape[0]
        inputs[i,:L] = X
        targets[i,:L] = y
        mask[i,:L] = 1.0
    inputs = torch.tensor(inputs, device=device)
    targets = torch.tensor(targets, device=device)
    mask = torch.tensor(mask, device=device)
    return {'inputs': inputs, 'targets': targets, 'mask': mask, 'lengths': lengths}


class TinyBehaviorRNNTrainer:
    """Trainer for TinyBehaviorRNN using supervised NLL + early stopping.

    Args:
        model: TinyBehaviorRNN instance
        lr: learning rate (Adam)
        weight_decay: L2 regularization coefficient
        max_epochs: maximum passes over training data
        batch_size: number of sessions per optimization step
        patience: early stopping patience based on validation NLL
        grad_clip: gradient norm clip (None disables)
    """
    def __init__(self, model: TinyBehaviorRNN, lr=5e-3, weight_decay=5e-4,
                 max_epochs=500, batch_size=16, patience=30, grad_clip=5.0,
                 device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.grad_clip = grad_clip
        self.history = {'train_nll': [], 'val_nll': []}

    def _epoch_batches(self, sessions):
        idx = np.arange(len(sessions))
        np.random.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch_ids = idx[start:start+self.batch_size]
            yield [sessions[i] for i in batch_ids]

    def _compute_nll(self, batch):
        logits, _ = self.model(batch['inputs'])  # (B,T,A)
        log_probs = F.log_softmax(logits, dim=-1)
        B,T,A = log_probs.shape
        gather = log_probs.gather(-1, batch['targets'].unsqueeze(-1)).squeeze(-1)
        masked = gather * batch['mask']
        nll = -masked.sum() / batch['mask'].sum().clamp_min(1.0)
        return nll

    def fit(self, train_sessions, val_sessions):
        best_val = float('inf')
        best_state = None
        epochs_no_improve = 0
        for epoch in range(1, self.max_epochs+1):
            self.model.train()
            train_losses = []
            for batch_sessions in self._epoch_batches(train_sessions):
                batch = collate_sessions(batch_sessions, self.model.num_actions, device=self.device)
                loss = self._compute_nll(batch)
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                train_losses.append(float(loss.item()))

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_batch = collate_sessions(val_sessions, self.model.num_actions, device=self.device)
                val_loss = self._compute_nll(val_batch)
            mean_train = float(np.mean(train_losses))
            self.history['train_nll'].append(mean_train)
            self.history['val_nll'].append(float(val_loss.item()))

            if val_loss < best_val - 1e-6:
                best_val = float(val_loss.item())
                best_state = {k: v.cpu().clone() for k,v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    @torch.no_grad()
    def evaluate(self, test_sessions):
        self.model.eval()
        batch = collate_sessions(test_sessions, self.model.num_actions, device=self.device)
        nll = self._compute_nll(batch)
        return {'test_nll': float(nll.item())}

    def hidden_trajectories(self, session):
        """Return hidden state trajectory for a single session (numpy)."""
        self.model.eval()
        batch = collate_sessions([session], self.model.num_actions, device=self.device)
        with torch.no_grad():
            logits, h_n = self.model(batch['inputs'])
        return logits.squeeze(0).cpu().numpy()

            if mode.lower().startswith("s"):
                p = np.round(np.random.uniform(low, high, size=N), decimals=decimals)
                return np.vstack([p, 1 - p]).T
            else:  # Unstructured
                p1 = np.round(np.random.uniform(low, high, size=N), decimals=decimals)
                p2 = np.round(np.random.uniform(low, high, size=N), decimals=decimals)
                return np.vstack([p1, p2]).T
        elif isinstance(mode, (list, tuple, np.ndarray)):
            arr = np.asarray(mode)
            if arr.shape == (2,):
                return np.tile(arr, (N, 1))
            assert arr.shape == (N, 2)
            return arr
        else:
            raise ValueError("Unsupported reward prob mode")

    def train_env(
        self,
        mode="Structured",
        n_sessions=2000,
        n_trials=200,
        clip_grad=1.0,
        progress=True,
        belief_target_detach=True,
    ):
        self.model.train()
        all_records = []
        reward_probs_all = self._sample_reward_probs(mode, N=n_sessions)
        for s in tqdm(range(n_sessions), disable=not progress):
            rps = reward_probs_all[s]
            prev_action = None
            prev_reward = 0.0
            inputs = []
            actions = []
            rewards = []
            chosen_belief_logits = []

            hidden = None
            for t in range(n_trials):
                x_t = (
                    self._build_input(prev_action, prev_reward)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )  # (1,1,D)
                inputs.append(x_t)
                with torch.no_grad():
                    belief_logits, policy_logits, values, hidden = self.model(
                        x_t, hidden
                    )
                probs = torch.softmax(policy_logits.squeeze(0), dim=-1)  # (1,A)
                dist = torch.distributions.Categorical(probs)
                act = dist.sample().item()
                reward = 1.0 if random.random() < rps[act] else 0.0

                actions.append(act)
                rewards.append(reward)
                prev_action, prev_reward = act, reward
                all_records.append(
                    {
                        "session": s + 1,
                        "trial": t + 1,
                        "action": act + 1,
                        "reward": reward,
                        "p_arm1": float(rps[0]),
                        "p_arm2": float(rps[1]),
                    }
                )

            # Build tensor sequence for learning pass
            x_seq = torch.cat(inputs, dim=1)  # (1,T,D)
            belief_logits_seq, policy_logits_seq, value_seq, _ = self.model(x_seq)
            policy_logits_seq = policy_logits_seq.squeeze(0)  # (T,A)
            belief_logits_seq = belief_logits_seq.squeeze(0)
            value_seq = value_seq.squeeze(0)

            # Compute returns & advantages
            G = self._discount(rewards)
            advantage = G - value_seq.detach()
            action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
            log_probs = torch.distributions.Categorical(
                logits=policy_logits_seq
            ).log_prob(action_tensor)
            policy_loss = -(log_probs * advantage).mean()
            value_loss = self.value_weight * (G - value_seq).pow(2).mean()
            entropy = (
                torch.distributions.Categorical(logits=policy_logits_seq)
                .entropy()
                .mean()
            )
            entropy_term = -self.entropy_weight * entropy

            # Belief calibration: for each trial, take predicted prob of reward for chosen action vs actual reward
            # Convert chosen arm belief logits to prob via sigmoid for binary reward.
            chosen_logits = belief_logits_seq[torch.arange(len(actions)), action_tensor]
            reward_targets = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            )
            if belief_target_detach:
                belief_loss = self.belief_weight * F.binary_cross_entropy_with_logits(
                    chosen_logits, reward_targets.detach()
                )
            else:
                belief_loss = self.belief_weight * F.binary_cross_entropy_with_logits(
                    chosen_logits, reward_targets
                )

            loss = policy_loss + value_loss + entropy_term + belief_loss
            self.optimizer.zero_grad()
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.optimizer.step()

            self.train_history.append(
                {
                    "session": s + 1,
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "belief_loss": float(belief_loss.item()),
                    "entropy": float(entropy.item()),
                }
            )

        return pd.DataFrame(all_records)

    @torch.no_grad()
    def evaluate(self, reward_probs, n_trials=200, greedy=True):
        self.model.eval()
        rps = np.asarray(reward_probs, dtype=float)
        prev_action = None
        prev_reward = 0.0
        hidden = None
        actions, rewards = [], []
        for t in range(n_trials):
            x_t = self._build_input(prev_action, prev_reward).unsqueeze(0).unsqueeze(0)
            belief_logits, policy_logits, values, hidden = self.model(x_t, hidden)
            probs = torch.softmax(policy_logits.squeeze(0), dim=-1)
            if greedy:
                act = int(torch.argmax(probs).item())
            else:
                dist = torch.distributions.Categorical(probs)
                act = dist.sample().item()
            reward = 1.0 if random.random() < rps[act] else 0.0
            actions.append(act)
            rewards.append(reward)
            prev_action, prev_reward = act, reward
        return {
            "actions": np.array(actions),
            "rewards": np.array(rewards),
        }
