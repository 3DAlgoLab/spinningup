import torch.nn as nn 
import torch 

# policy, deterministic
def mu(state_t):
    return 'action'

# stochastic policy
def phi(state_t):
    return 'action'



obs_dim = 10 
act_dim = 3
pi_net = nn.Sequential(nn.Linear(obs_dim, 64), 
                       nn.Tanh(), 
                       nn.Linear(64, 64), 
                       nn.Tanh(), 
                       nn.Linear(64, act_dim))

torch.normal 