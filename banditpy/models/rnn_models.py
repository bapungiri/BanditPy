import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os
import math  # Ensure math is imported for PaperBanditLSTM


# Original BanditLSTMModel (can be kept or removed if PaperBanditLSTM is the replacement)
class BanditLSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=48, output_size=2):
        super(BanditLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_q = nn.Linear(hidden_size, output_size)
        self.fc_v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        q = self.fc_q(out)
        v = self.fc_v(out)
        return q, v.squeeze(-1)


# LSTM Model based on the paper's description
class PaperBanditLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=48, num_actions=2):
        super(PaperBanditLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_q = nn.Linear(hidden_size, num_actions)
        self.fc_v = nn.Linear(hidden_size, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

        nn.init.normal_(
            self.fc_q.weight.data, mean=0.0, std=1.0 / math.sqrt(self.hidden_size)
        )
        nn.init.constant_(self.fc_q.bias.data, 0)

        nn.init.normal_(
            self.fc_v.weight.data, mean=0.0, std=1.0 / math.sqrt(self.hidden_size)
        )
        nn.init.constant_(self.fc_v.bias.data, 0)

    def forward(self, x, hidden_state=None):
        lstm_out, new_hidden_state = self.lstm(
            x, hidden_state
        )  # new_hidden_state is (h_n, c_n)
        q_values = self.fc_q(lstm_out)
        v_value_raw = self.fc_v(lstm_out)
        v_value = v_value_raw.squeeze(-1)
        return q_values, v_value, new_hidden_state  # Return new_hidden_state


class BanditTrainer:
    def __init__(
        self,
        n_sessions=100,
        trials_per_session=200,
        structured=True,
        lr=1e-3,
        gamma=0.8,
        beta_entropy=0.5,
        beta_value=0.5,
        model_path="bandit_model_paper.pt",  # Adjusted model path name
        hidden_size_lstm=48,
    ):
        self.n_sessions = n_sessions
        self.trials_per_session = trials_per_session
        self.structured = structured
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.beta_value = beta_value
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PaperBanditLSTM(hidden_size=hidden_size_lstm).to(self.device)
        # self.model = BanditLSTMModel(hidden_size=hidden_size_lstm).to(self.device)
        # Optimizer as per paper: RMSprop with decay rate 0.99
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.99)
        self.training_loss = []

    def _get_reward_probs(self):
        probs = np.arange(0.1, 1.0, 0.1)
        if self.structured:
            p1 = np.random.choice(probs)
            p2 = 1.0 - p1
        else:
            p1, p2 = np.random.choice(probs, 2)
        return [p1, p2]

    def _generate_input(self, action, reward):
        # Input: one-hot action (2 units) + reward (1 unit) = 3 units
        input_vec = torch.zeros(3)
        input_vec[action] = 1  # Assumes action is 0 or 1
        input_vec[2] = reward  # Last element for reward
        return input_vec.to(self.device)

    def _discounted_return(self, rewards):
        # R(t) = sum_{i=t}^{k-1} gamma^(i-t) * r_{i+1} (paper uses r_i for reward at step i)
        # Correcting to use r_i for reward at step i, so R_t = r_t + gamma*r_{t+1} + ...
        # The loop calculates G_t = r_t + gamma * G_{t+1}
        G = []
        R_val = 0.0  # Ensure float for calculation
        # Iterating from the last reward backwards
        for r in reversed(rewards):
            R_val = r + self.gamma * R_val
            G.insert(0, R_val)
        return torch.tensor(G, dtype=torch.float32).to(self.device)

    def train(self):
        results = []
        for session_id in range(self.n_sessions):
            reward_probs = self._get_reward_probs()

            input_seq_for_update, actions, rewards = [], [], []
            trial_logs = []

            current_input_for_action_selection = torch.zeros((1, 1, 3)).to(self.device)
            # Initialize hidden state for action selection per session
            hidden_for_action_selection = None

            for trial_id in range(self.trials_per_session):
                # Pass and update hidden state for action selection
                q_vals_step, _, hidden_for_action_selection = self.model(
                    current_input_for_action_selection, hidden_for_action_selection
                )  # q_vals_step: [1, 1, num_actions]

                p_step = F.softmax(q_vals_step.squeeze(0), dim=-1)
                dist_step = torch.distributions.Categorical(p_step)
                action = dist_step.sample().item()

                reward = 1.0 if random.random() < reward_probs[action] else 0.0

                actions.append(action)
                rewards.append(reward)

                next_input_tensor = self._generate_input(action, reward)
                input_seq_for_update.append(next_input_tensor.cpu())

                current_input_for_action_selection = next_input_tensor.unsqueeze(
                    0
                ).unsqueeze(0)

                trial_logs.append(
                    {
                        "session_id": session_id,
                        "trial_id": trial_id,
                        "chosen_action": action,
                        "reward": reward,
                        "reward_prob": reward_probs[action],
                        "model_prediction": p_step.squeeze().cpu().tolist(),
                    }
                )

            G = self._discounted_return(rewards)

            x_seq_tensor = (
                torch.stack(input_seq_for_update).unsqueeze(0).to(self.device)
            )

            # For full sequence, hidden state is managed internally by LSTM for the sequence
            # Or, we can pass the initial hidden_for_action_selection if we want continuity,
            # but typically for batch updates, each sequence in a batch starts fresh or with a learned h0.
            # Here, since it's one full sequence, starting fresh (hidden_state=None) is fine.
            q_vals_seq, v_vals_seq, _ = self.model(
                x_seq_tensor
            )  # Unpack the third returned value (hidden_state)

            q_vals_seq = q_vals_seq.squeeze(0)
            v_vals_seq = v_vals_seq.squeeze(0)

            # ...existing code...
            dist_entropy = (
                torch.distributions.Categorical(logits=q_vals_seq).entropy().mean()
            )
            actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
            log_probs = torch.distributions.Categorical(logits=q_vals_seq).log_prob(
                actions_tensor
            )
            advantage = G - v_vals_seq
            policy_loss = -(log_probs * advantage.detach()).mean()
            value_loss = self.beta_value * advantage.pow(2).mean()
            entropy_bonus = -self.beta_entropy * dist_entropy
            loss = policy_loss + value_loss + entropy_bonus

            self.training_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            results.extend(trial_logs)
            if (session_id + 1) % 10 == 0:  # Log progress
                print(
                    f"Session {session_id+1}/{self.n_sessions}, Avg Loss: {np.mean(self.training_loss[-10:]) :.4f}"
                )

        df = pd.DataFrame(results)
        df.to_csv("bandit_results_paper_model.csv", index=False)
        print(f"Training complete. Results saved to bandit_results_paper_model.csv")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.eval()
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

    def evaluate(self, p1, p2):
        try:  # Add try-except for robust model loading
            self.load_model()
        except FileNotFoundError as e:
            print(f"Error loading model for evaluation: {e}")
            print(
                "Please ensure the model is trained and saved, or check the model_path."
            )
            return  # Exit if model can't be loaded

        self.model.eval()

        all_eval_results = []
        print(f"Starting evaluation for {p1,p2} sessions...")

        # for session_id in range(n_sessions):
        # reward_probs = self._get_reward_probs()
        reward_probs = [p1, p2]
        session_trial_logs = []

        current_input_for_eval = torch.zeros((1, 1, 3)).to(self.device)
        hidden_eval = None  # Initialize hidden state for the session

        for trial_id in range(self.trials_per_session):
            with torch.no_grad():
                # Pass hidden_eval and get the new one
                q_vals_step, _, hidden_eval = self.model(
                    current_input_for_eval, hidden_eval
                )

            p_step = F.softmax(q_vals_step.squeeze(0), dim=-1)
            action = torch.argmax(p_step).item()

            reward = 1.0 if random.random() < reward_probs[action] else 0.0

            next_input_tensor = self._generate_input(action, reward)
            current_input_for_eval = next_input_tensor.unsqueeze(0).unsqueeze(0)

            session_trial_logs.append(
                {
                    # "session_id": session_id,
                    "trial_id": trial_id,
                    "chosen_action": action,
                    "reward": reward,
                    "reward_prob": reward_probs[action],
                    "model_prediction": p_step.squeeze().cpu().tolist(),
                }
            )
        all_eval_results.extend(session_trial_logs)
        if len(session_trial_logs) > 0:  # Avoid error if trials_per_session is 0
            print(
                f"Evaluation Session complete. Avg reward: {np.mean([log['reward'] for log in session_trial_logs]):.2f}"
            )
        else:
            print(f"Evaluation Session complete. No trials to average reward.")

        df_eval = pd.DataFrame(all_eval_results)
        if not df_eval.empty:
            df_eval.to_csv("bandit_eval_results_paper_model.csv", index=False)
            print(
                f"Evaluation complete. Results saved to bandit_eval_results_paper_model.csv"
            )
        else:
            print(f"Evaluation complete. No results to save.")

    def get_model_weights(self):
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
