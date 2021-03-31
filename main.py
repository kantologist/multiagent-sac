""" Packaged MASAC """
import argparse
import torch
import numpy as np
from unityagents import UnityEnvironment
from agents.dqn_agent import DQNAgent
from agents.sac_agent import SacAgent
from agents.masac_agent import MaSacAgent

def seed_torch(seed):
    torch.manual_seed = seed
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main():
    env = UnityEnvironment(file_name="TennisEnvironment/Tennis.app")
    seed = 777
    np.random.seed(seed)
    seed_torch(seed)

    num_episode = 1000
    memory_size = 10000
    batch_size = 64

    agent = MaSacAgent(env, memory_size, batch_size)
    agent.train(num_episode)

    agent.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Rl algorithms')
    parser.add_argument('-m', '--mode', type= str, default="multiagent", 
                        choices=["multiagent"],
                        help='determines which algorithm to train')

    args = parser.parse_args()
    if args.mode == "multiagent":
        main()
    else:
        main()
