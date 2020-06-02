from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
USE_CUDA = torch.cuda.is_available()

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            ######## YOUR CODE HERE! ########



            # get the vector of q values (max CDR for each possible action)
            q_value = self.forward(state)

            # get the action that has the maximum q value (highest of the CDRs)
            action = torch.argmax(q_value)



            ######## YOUR CODE HERE! ########
        else:
            action = random.randrange(self.env.action_space.n)
        return action
        
def compute_td_loss(model, batch_size, gamma, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    # print("state {}: {}".format(state.size(), state))
    # print("next state {}: {}".format(next_state.size(), next_state))
    # print("Action {}: {}".format(action.size(), action))
    # print("reward {}: {}".format(reward.size(), reward))
    # print("done {}: {}".format(done.size(), done))

    ######## YOUR CODE HERE! ########
    # TODO: Implement the Temporal Difference Loss
    





    # this is taking the state and querying the prediction network 
    # to see what q values it gives for each possible action given
    # that we are in this particular state
    q_this_state_predicted, _ = model.forward(state).max(1)
    next_q_values, _ = model.forward(next_state).max(1)
    q_this_state_target = reward + ( gamma * next_q_values )


    # print("q_this_state_target {}: {}".format(q_this_state_target.size(), q_this_state_target))
    # print("q_this_state_predicted {}: {}".format(q_this_state_predicted.size(), q_this_state_predicted))

    loss = (q_this_state_target - q_this_state_predicted).pow(2).sum() # do i need to divide by batch_size?

    # print("loss: {}".format(loss))




    ######## YOUR CODE HERE! ########
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        ######## YOUR CODE HERE! ########




        samples = random.sample(self.buffer, batch_size)
        state = [ x[0] for x in samples ]
        action = [ x[1] for x in samples ]
        reward = [ x[2] for x in samples ]
        next_state = [ x[3] for x in samples ]
        done = [ x[4] for x in samples ]




        ######## YOUR CODE HERE! ########
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
