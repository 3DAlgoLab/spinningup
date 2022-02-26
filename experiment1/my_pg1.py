import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
import numpy as np


def mlp(sizes, activation=nn.ReLU6, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1], act())]

    return nn.Sequential(*layers)


def list_to_tensor(list_arr, dtype=torch.float32):
    return torch.tensor(np.array(list_arr), dtype=dtype)


def train(env_name, hidden_sizes=[32], lr=1e-2, epochs=50,
          batch_size=5000, render=False):
    assert env_name

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

    # action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    optimizer = Adam(logits_net.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        # reset episode-specific variables
        obs = env.reset()
        done = False
        ep_rews = []  # list for rewards accrued throughout ep.

        finished_rending_this_epoch = False
        while True:
            if not finished_rending_this_epoch and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())
            # act
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []
                finished_rending_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()

        if i == 5:  # TODO: remove after check
            breakpoint()
        batch_loss = compute_loss(obs=list_to_tensor(batch_obs),
                                  act=list_to_tensor(batch_acts, dtype=torch.int32),
                                  weights=list_to_tensor(batch_weights))
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print(f"epoch: {i:3d}\t loss:{batch_loss:.3f}\t"
              f"return: {np.mean(batch_rets):.3f}\t ep_len: {np.mean(batch_lens):.3f}\t")


if __name__ == "__main__":
    # Test
    # m = Categorical(torch.tensor([1., 1, 1, 1, 1]))
    # for i in range(10):
    #     r = m.sample()
    #     print(r)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print("\nSimplest PG")
    train(env_name=args.env_name, render=args.render, lr=args.lr)
