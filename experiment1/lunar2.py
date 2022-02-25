from spinup import ppo_pytorch as ppo
import gym
from torch.nn import ReLU
from spinup.utils.mpi_tools import mpi_fork


def env_fn(): return gym.make('LunarLander-v2')


ac_kwargs = dict(hidden_sizes=[64, 64], activation=ReLU)
logger_kwargs = dict(output_dir='./output', exp_name='ppo_torch2')


if __name__ == "__main__":
    # mpi_fork(8)
    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000,
        epochs=250, logger_kwargs=logger_kwargs)
