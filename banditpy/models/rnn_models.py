import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os
import math  # Ensure math is imported for PaperBanditLSTM


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
        # Task specific defaults
        n_train_sessions=10000,
        n_test_sessions=300,
        trials_per_session=100,
        # Model and training hparams
        input_size=3,  # 2 for one-hot action (1,2) + 1 for reward
        hidden_size=48,
        num_model_actions=2,  # Model outputs Q-values for 2 actions (0, 1)
        lr=0.0007,
        gamma=0.9,
        beta_entropy=0.05,
        beta_value=0.05,
        model_path="two_arm_task_model.pt",
        device=None,
    ):

        self.n_train_sessions = n_train_sessions
        self.n_test_sessions = n_test_sessions
        self.trials_per_session = trials_per_session
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.beta_value = beta_value
        self.model_path = model_path
        self.input_size = input_size
        self.num_model_actions = (
            num_model_actions  # Number of outputs from the policy head
        )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Using WangEtAlRNN as the model
        self.model = BanditLSTMModel(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_actions=self.num_model_actions,  # policy_head outputs this many logits
            recurrent_type="lstm",
        ).to(self.device)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.99)
        self.training_loss_history = []

    def _get_reward_probs(self, structured):
        """
        Generates reward probabilities for the two arms for a session.
        """

        if structured:
            p_arm1 = np.random.uniform(0, 1)
            p_arm2 = 1.0 - p_arm1
        else:
            p_arm1, p_arm2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
        return [p_arm1, p_arm2]  # Index 0 for arm 1, index 1 for arm 2

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

    def train(self, structured=True):
        print(f"Starting training for {self.n_train_sessions} sessions...")
        training_data = []
        for session_idx in range(self.n_train_sessions):
            session_reward_probs = self._get_reward_probs(structured)

            input_tensors_for_update, model_actions_taken, rewards_received = [], [], []

            current_input_for_model = torch.zeros(
                (1, 1, self.input_size), device=self.device
            )
            lstm_hidden_state = None

            for _ in range(self.trials_per_session):
                policy_logits_step, _, lstm_hidden_state = self.model(
                    current_input_for_model, lstm_hidden_state
                )

                prob_step = F.softmax(policy_logits_step.squeeze(0), dim=-1)
                dist_step = torch.distributions.Categorical(prob_step)
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

            self.training_loss_history.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if (session_idx + 1) % 100 == 0:
                avg_loss = (
                    np.mean(self.training_loss_history[-100:])
                    if self.training_loss_history
                    else float("nan")
                )
                print(
                    f"Training Session {session_idx+1}/{self.n_train_sessions}, Avg Loss (last 100): {avg_loss:.4f}"
                )

        final_avg_loss = (
            np.mean(self.training_loss_history[-100:])
            if self.training_loss_history
            else float("nan")
        )
        print(f"Training complete. Final avg loss: {final_avg_loss:.4f}")
        self.save_model()

        df_training_results = pd.DataFrame(training_data)
        return df_training_results

    def evaluate(self, reward_probs=None):
        print("Starting evaluation with fixed weights...")
        try:
            self.load_model()  # Loads model and sets to eval mode
        except FileNotFoundError:
            print(f"Evaluation failed: Model file not found at {self.model_path}.")
            return pd.DataFrame()

        evaluation_data = []

        for session_idx in range(self.n_test_sessions):
            # session_reward_probs = self._get_reward_probs(structured)
            session_reward_probs = reward_probs

            current_input_for_model = torch.zeros(
                (1, 1, self.input_size), device=self.device
            )
            lstm_hidden_state = None

            for _ in range(self.trials_per_session):
                with torch.no_grad():
                    policy_logits_step, _, lstm_hidden_state = self.model(
                        current_input_for_model, lstm_hidden_state
                    )

                prob_step = F.softmax(policy_logits_step.squeeze(0), dim=-1)
                # model_action = torch.argmax(prob_step).item()  # Greedy action (0 or 1)
                dist_step = torch.distributions.Categorical(prob_step)
                model_action = dist_step.sample().item()

                env_action = model_action + 1  # Convert to 1 or 2

                reward = (
                    1.0 if random.random() < session_reward_probs[model_action] else 0.0
                )

                evaluation_data.append(
                    {
                        "session_id": session_idx + 1,
                        "chosen_action": env_action,
                        "reward": reward,
                        "arm1_reward_prob": session_reward_probs[0],
                        "arm2_reward_prob": session_reward_probs[1],
                    }
                )

                next_input_tensor = self._generate_input(env_action, reward)
                current_input_for_model = next_input_tensor.unsqueeze(0).unsqueeze(0)

            if (session_idx + 1) % 50 == 0:
                print(
                    f"Evaluation Session {session_idx+1}/{self.n_test_sessions} complete."
                )

        df_evaluation_results = pd.DataFrame(evaluation_data)
        print("Evaluation complete.")
        return df_evaluation_results

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.eval()  # Ensure model is in evaluation mode after loading
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

    def get_model_weights(self):
        """
        Returns the model weights as a dictionary.
        """
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
