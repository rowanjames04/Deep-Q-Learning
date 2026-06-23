import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from experiment_replay import ReplayMemory
import itertools
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu' #this can probably be deleted later

class Agent:

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
    
    def run(self, is_training=True, render=False):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []
        
        policy_dqn = DQN(num_states, num_actions).to_device(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
        

        for episode in itertools.count():
            state, _ = env.reset()
            terminated = False
            episode_reward = 0.0

            while not terminated:
                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample() #sample function pulls random action == 0 -> do nothing, 1 -> flap

                # Processing:
                new_state, reward, terminated, _, info = env.step(action) #executes action
                
                episode_reward += reward

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                state = new_state

            rewards_per_episode.append(episode_reward)