""" Packaged DRLND """
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

def navigation_main():
    env = UnityEnvironment(file_name="Navigation/Banana.app")
    seed = 777
    np.random.seed(seed)
    seed_torch(seed)

    num_episode = 2000
    memory_size = 10000
    batch_size = 64
    target_update = 4
    epsilon_decay = 0.9

    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
    agent.train(num_episode)

    agent.test()

def continuous_main():
    env = UnityEnvironment(file_name="Continuous Control/Reacher.app")
    seed = 777
    np.random.seed(seed)
    seed_torch(seed)

    num_episode = 300
    memory_size = 10000
    batch_size = 64

    agent = SacAgent(env, memory_size, batch_size)
    agent.train(num_episode)

    agent.test()

def ma_main():
    env = UnityEnvironment(file_name="Collaboration and Competition/Tennis.app")
    seed = 777
    np.random.seed(seed)
    seed_torch(seed)

    num_episode = 1500
    memory_size = 10000
    batch_size = 64

    agent = MaSacAgent(env, memory_size, batch_size)
    agent.train(num_episode)

    agent.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Rl algorithms')
    parser.add_argument('-m', '--mode', type= str, default="navigation", 
                        choices=["navigation", "continuous", "multiagent"],
                        help='determines which algorithm to train')

    args = parser.parse_args()
    if args.mode == "continuous":
        continuous_main()
    elif args.mode == "multiagent":
        ma_main()
    else:
        navigation_main()
