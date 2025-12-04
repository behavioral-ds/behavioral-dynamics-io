import numpy as np

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
from imitation.rewards.reward_nets import RewardNet

class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_states):
        super(OneHotWrapper, self).__init__(env)
        self.n_states = n_states
        # Define a Box space with the one-hot representation.
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_states,), dtype=np.float32)
    
    def observation(self, observation):
        onehot = np.zeros(self.n_states, dtype=np.float32)
        onehot[observation] = 1.0
        return onehot
    
    # Override reset to return (obs, info)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    # Override step to process the observation and pass through the other outputs.
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

class OneHotRewardNet(RewardNet):
    def __init__(self, observation_space, action_space, nodes, normalize_input_layer=None, **kwargs):
    # def __init__(self, observation_space, action_space, normalize_input_layer=None, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        # Define the network architecture for one-hot encoded observations and actions.
        self.obs_dim = observation_space.shape[0]  # Size of the one-hot encoded state
        self.act_dim = action_space.n  # Number of actions (assuming discrete actions)
        
        # Preprocessing layers for observations and actions.
        self.preprocess_obs = nn.Sequential(
            nn.Linear(self.obs_dim, nodes),
            nn.ReLU(),
        )
        self.preprocess_act = nn.Sequential(
            nn.Linear(self.act_dim, nodes),
            nn.ReLU(),
        )
        
        # Combined layer to process the concatenated state and action.
        self.combined_layer = nn.Sequential(
            nn.Linear(nodes * 2, nodes),  # 16 (state) + 16 (action) = 32
            nn.ReLU(),
            nn.Linear(nodes, 1),  # Output a scalar reward
        )
        
        # Handle input normalization.
        if normalize_input_layer is not None:
            self.obs_normalize = normalize_input_layer(self.obs_dim)
            self.act_normalize = normalize_input_layer(self.act_dim)
        else:
            self.obs_normalize = nn.Identity()
            self.act_normalize = nn.Identity()

    def forward(self, state, action, next_state, done):
        batch_size = state.shape[0]  # Get batch size

        # If action is one-hot encoded (shape: [batch_size, n_actions]), convert it to an index
        if action.shape[1] == self.act_dim:  # Already one-hot encoded
            action = action.argmax(dim=1)  # Convert from one-hot to discrete action indices

        # Create a one-hot encoded action tensor
        one_hot_action = torch.zeros((batch_size, self.act_dim), dtype=torch.float32, device=action.device)

        # Scatter requires action to be (batch_size, 1)
        one_hot_action.scatter_(1, action.unsqueeze(1), 1.0)  # Convert discrete actions to one-hot

        # Normalize inputs
        state = self.obs_normalize(state)
        one_hot_action = self.act_normalize(one_hot_action)

        # Pass through network
        state_features = self.preprocess_obs(state)
        action_features = self.preprocess_act(one_hot_action)

        combined = torch.cat([state_features, action_features], dim=-1)
        reward = self.combined_layer(combined)
        reward = reward.squeeze(-1)

        return reward

# A simple custom Gym environment using your transition probabilities.
class RedditEnv(gym.Env):
    def __init__(self, n_states, n_actions, tp):
        super(RedditEnv, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.tp = tp  # Transition probability matrix: shape [n_states, n_actions, n_states]
        # Although we originally defined the observation_space as Discrete,
        # the OneHotWrapper will convert observations to a Box.
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Discrete(n_actions)
        self.state = 0  # Starting state
        self.render_mode = None  # Define render_mode to silence warning
        self.num_envs = 1       # Optionally, add num_envs if required
        
    # Updated reset method: returns (observation, info) as per Gymnasium API.
    def reset(self, **kwargs):
        self.state = 0  # or choose a random starting state
        return self.state, {}
    
    # Updated step method: returns (observation, reward, terminated, truncated, info)
    def step(self, action):
        next_state = np.random.choice(np.arange(self.n_states), p=self.tp[self.state, action])
        reward = 0  # Reward will be provided by the discriminator in GAIL.
        terminated = False  # Set to True if episode ends due to task success/failure.
        truncated = False   # Set to True if episode ends due to a time limit.
        self.state = next_state
        return self.state, reward, terminated, truncated, {}