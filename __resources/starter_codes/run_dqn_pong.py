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

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 1500000
batch_size = 32
gamma = 0.9
    
replay_initial = 10000
replay_buffer = ReplayBuffer(1000000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
if USE_CUDA:
    model = model.cuda()

epsilon_start = 0.1
epsilon_final = 0.1
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()

if len(sys.argv) > 1:
    checkpoint = torch.load(sys.argv[1])
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    #frame_start = checkpoint['frame_idx']
    #losses = checkpoint['losses']
    #all_rewards = checkpoint['all_rewards']
    #replay_buffer = checkpoint['replay_buffer']

frame_start = 1300000

for frame_idx in range(frame_start, frame_start + num_frames + 1):

    epsilon = epsilon_by_frame(frame_idx)

    # given our state (received from the env), model chooses an action
    action = model.act(state, epsilon)
    
    # query environment for resulting reward and next state; save in replay buffer
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.cpu().numpy())

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: {}, preparing replay buffer'.format(frame_idx))

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('#Frame: {}, Loss: {}'.format(frame_idx, np.mean(losses)))
        print('Last-10 average reward: {}'.format(np.mean(all_rewards[-10:])))

    if frame_idx % 500000 == 0:
        checkpoint = {
            'state_dict'    : model.state_dict(),
            'optimizer'     : optimizer.state_dict(),
            'frame_idx'     : frame_idx,
            'losses'        : losses,
            'all_rewards'   : all_rewards,
            # 'replay_buffer' : replay_buffer
        }

        torch.save(checkpoint, "./checkpoints/{}_checkpoint".format(frame_idx))
