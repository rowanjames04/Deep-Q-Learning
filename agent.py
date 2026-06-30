import os
import argparse
import itertools
import random
from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')  # render plots to image files, not to screen (must precede pyplot import)
import matplotlib.pyplot as plt

import torch
from torch import nn
import yaml

import flappy_bird_gymnasium  # noqa: F401  registers the FlappyBird envs with gymnasium

from experience_replay import ReplayMemory
from dqn import DQN

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Deep Q-Learning Agent
class Agent():

    def __init__(self, hyperparameter_set):
        with open(os.path.join(BASE_DIR, 'hyperparameters.yml'), 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # steps before syncing policy => target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # training batch sampled from replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this reward
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.dueling_hidden_dim = hyperparameters.get('dueling_hidden_dim', 256)
        self.env_make_params    = hyperparameters.get('env_make_params', {})  # optional env-specific params
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']        # double dqn on/off flag
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']       # dueling dqn on/off flag
        self.seed               = hyperparameters.get('seed')                 # optional rng seed

        # Device: 'auto' uses CUDA if available else CPU; 'cpu'/'cuda' forces it.
        device = hyperparameters.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Neural network
        self.loss_fn = nn.MSELoss()   # NN loss function; swap for Huber/SmoothL1 to reduce outlier sensitivity
        self.optimizer = None         # initialised in run()

        # Path to run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False):
        os.makedirs(RUNS_DIR, exist_ok=True)

        if is_training:
            self._set_seeds()
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = self._make_env(render)
        try:
            num_actions = env.action_space.n
            num_states = env.observation_space.shape[0]  # Box(low, high, (shape0,), float64)

            if is_training and self.seed is not None:
                env.action_space.seed(self.seed)

            rewards_per_episode = []

            policy_dqn = DQN(num_states, num_actions, self.fc1_nodes,
                             self.enable_dueling_dqn, self.dueling_hidden_dim).to(self.device)

            if is_training:
                # Initialize epsilon
                epsilon = self.epsilon_init

                # Initialize replay memory
                memory = ReplayMemory(self.replay_memory_size, seed=self.seed)

                # Target network, identical to policy network initially
                target_dqn = DQN(num_states, num_actions, self.fc1_nodes,
                                 self.enable_dueling_dqn, self.dueling_hidden_dim).to(self.device)
                target_dqn.load_state_dict(policy_dqn.state_dict())

                # Policy network optimizer
                self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

                epsilon_history = []
                step_count = 0          # steps taken since last target sync
                best_reward = -float('inf')
                seeded = False
            else:
                # Load learned policy and switch to evaluation mode
                policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=self.device))
                policy_dqn.eval()

            # Train indefinitely; stop manually when satisfied with results.
            for episode in itertools.count():

                # Seed only the first reset so episodes stay stochastic after the first.
                reset_seed = self.seed if (is_training and self.seed is not None and not seeded) else None
                state, _ = env.reset(seed=reset_seed)
                seeded = True
                state = torch.tensor(state, dtype=torch.float, device=self.device)

                terminated = False
                episode_reward = 0.0

                # Act until the episode terminates or the reward cap is hit
                # (some envs let a trained agent run forever, so the cap is a safety valve).
                while not terminated and episode_reward < self.stop_on_reward:
                    action = self._select_action(state, epsilon if is_training else 0.0,
                                                 policy_dqn, env, is_training)

                    new_state, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward

                    new_state_t = torch.tensor(new_state, dtype=torch.float, device=self.device)

                    if is_training:
                        # Store CPU copies + a raw float reward so the replay buffer
                        # doesn't hold device tensors (which would pin GPU memory).
                        memory.append((
                            state.cpu(),
                            torch.tensor(action, dtype=torch.int64),
                            new_state_t.cpu(),
                            float(reward),
                            terminated,
                        ))
                        step_count += 1

                    state = new_state_t

                rewards_per_episode.append(episode_reward)

                if not is_training:
                    continue

                best_reward = self._maybe_save_best(episode, episode_reward, best_reward, policy_dqn)

                # Refresh the plot at most every 10 seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # Learn once enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon (per optimisation step)
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Sync target network periodically
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
        finally:
            env.close()

    def _make_env(self, render):
        # **self.env_make_params passes env-specific options from hyperparameters.yml.
        return gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

    def _select_action(self, state, epsilon, policy_dqn, env, is_training):
        # Epsilon-greedy: explore with a random action, else exploit the policy
        if is_training and random.random() < epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            # Add batch dim (tensor([1,2,3]) -> tensor([[1,2,3]])), argmax over actions, squeeze back.
            return policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()

    def _maybe_save_best(self, episode, episode_reward, best_reward, policy_dqn):
        if episode_reward > best_reward:
            if best_reward == -float('inf'):
                log_message = (f"{datetime.now().strftime(DATE_FORMAT)}: New best reward "
                               f"{episode_reward:0.1f} at episode {episode}, saving model...")
            else:
                delta_pct = (episode_reward - best_reward) / abs(best_reward) * 100
                log_message = (f"{datetime.now().strftime(DATE_FORMAT)}: New best reward "
                               f"{episode_reward:0.1f} ({delta_pct:+.1f}%) at episode {episode}, saving model...")
            print(log_message)
            with open(self.LOG_FILE, 'a') as file:
                file.write(log_message + '\n')
            torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
            return episode_reward
        return best_reward

    def _set_seeds(self):
        if self.seed is None:
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Mean reward (100-episode rolling window) vs episodes
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        plt.subplot(121)  # 1x2 grid, cell 1
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Epsilon decay vs optimisation steps
        plt.subplot(122)  # 1x2 grid, cell 2
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack the CPU tensors stored in memory and move the batch to the device
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        new_states = torch.stack(new_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        terminations = torch.tensor(terminations, dtype=torch.float, device=self.device)

        with torch.no_grad():
            if self.enable_double_dqn:
                # Action chosen by policy, value evaluated by target => reduces overestimation
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # Standard DQN target: max over actions from the target network
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        # Q values of the actions actually taken, from the current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute loss and update (backpropagation)
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()  # clear gradients
        loss.backward()             # compute gradients
        self.optimizer.step()       # update weights


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='hyperparameter set name from hyperparameters.yml')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)