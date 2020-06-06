from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer
import sys

# initialize the game environment
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

# initialize learner with the same parameters that it was trained with
batch_size = 32
gamma = 0.99
replay_buffer = ReplayBuffer(100000)
model = QLearner(env, 5000000, batch_size, gamma, replay_buffer)
if USE_CUDA:
    model = model.cuda()

losses = []
all_rewards = []
episode_reward = 0

# load the saved model
if len(sys.argv) > 1:
    checkpoint = torch.load(sys.argv[1])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

# no random actions during this eval stage
epsilon = -1

num_frames = 30000

state = env.reset()

for frame_idx in range(1, 1 + num_frames):

    # given our state (received from the env), model chooses an action
    action = model.act(state, epsilon)
    
    # query environment for resulting reward and next state; save in replay buffer
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        print(episode_reward)
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    # if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
    #     print('#Frame: {}, preparing replay buffer'.format(frame_idx))

    # if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
    #     print('#Frame: {}, Loss: {}'.format(frame_idx, np.mean(losses)))
    #     print('Last-10 average reward: {}'.format(np.mean(all_rewards[-10:])))

embeddings = []
samples = model.replay_buffer.sample(1000)

for sample in samples:
    embeddings.append(model.features(sample))

from sklearn.manifold import Isomap

x = sklearn.manifold.Isomap(n_components=2)
x.fit(embeddings)