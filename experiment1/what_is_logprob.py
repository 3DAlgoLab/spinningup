# %%
import torch
import torch.nn as nn
from torch.distributions import Categorical
# %%
action_logits = torch.rand(5)
action_probs = nn.Softmax(dim=-1)(action_logits)

print(action_probs)

# %%
sum(action_probs)
# %%
dist = Categorical(action_probs)
action = dist.sample()
# %%
# following two values are same

dist.log_prob(action)
# %%
torch.log(action_probs[action])

# %%
