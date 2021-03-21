""" Packaged Rainbow """
import torch
import numpy as np
from unityagents import UnityEnvironment
from agents.dqn_agent import DQNAgent

def seed_torch(seed):
    torch.manual_seed = seed
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def main():
    env = UnityEnvironment(file_name="Navigation/Banana.app")
    seed = 777
    np.random.seed(seed)
    seed_torch(seed)

    num_episode = 2000
    memory_size = 10000
    batch_size = 64
    target_update = 4
    epsilon_decay = 1 / 3000

    agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
    agent.train(num_episode)

    agent.test()

if __name__ == "__main__":
    main()