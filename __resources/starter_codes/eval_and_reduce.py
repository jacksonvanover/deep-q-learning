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
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
import hashlib

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

num_frames = 100000

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

# get 1000 samples from the replay buffer
state, action, reward, next_state, done = model.replay_buffer.sample(1000)

# save images
for i, image in enumerate(state):
    plt.imsave('./figures/{:04d}_state.png'.format(i), image.squeeze(), cmap='gray')

# convert to Tensors and find learned embedding
state = Variable(torch.FloatTensor(np.float32(state)))
embeddings = model.features(state)
embeddings = embeddings.view(embeddings.size(0), -1)
embeddings = model.fc[0](embeddings)

# dimensionality reduction
from sklearn.manifold import Isomap
x = Isomap(n_components=2)
x.fit(embeddings.cpu().detach().numpy())

data = np.concatenate([x.embedding_, [[x.item()] for x in action], [[x] for x in range(x.embedding_.shape[0])]], axis=1)
stay = np.array([x for x in data if x[2] == 0 or x[2] == 1])
right = np.array([x for x in data if x[2] == 2 or x[2] == 4])
left = np.array([x for x in data if x[2] == 3 or x[2] == 5])

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=stay[...,0],
        y=stay[...,1],
        mode='markers',
        name='Stay',
        text=stay[...,3]
    )
)
fig.add_trace(
    go.Scatter(
        x=right[...,0],
        y=right[...,1],
        mode='markers',
        name='Right',
        text=right[...,3]
    )
)
fig.add_trace(
    go.Scatter(
        x=left[...,0],
        y=left[...,1],
        mode='markers',
        name='Left',
        text=left[...,3]
    )
)
fig.write_html("./figures/dim_reduction.html")