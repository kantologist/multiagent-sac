""" Packaged DRLND"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from unityagents import UnityEnvironment
from buffers.buffer import ReplayBuffer
from models.network import Network
from torch.nn.utils.clip_grad import clip_grad_norm_

class DQNAgent:

    def __init__(
        self,
        env: UnityEnvironment,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float = 1 / 2000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        ):
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        env_info = env.reset(train_mode=True)[self.brain_name]
        self.env = env
        action_size = self.brain.vector_action_space_size
        state = env_info.vector_observations[0]
        state_size = len(state)
        
        self.obs_dim = state_size
        self.action_dim = 1

        self.memory = ReplayBuffer(self.obs_dim, self.action_dim, memory_size, batch_size)


        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.epsilon = max_epsilon

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.dqn = Network(self.obs_dim, self.action_dim)
        self.dqn_target = Network(self.obs_dim, self.action_dim)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=5e-5)

        self.transition = list()

        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.int64:
        """ Select an action given input """
        if self.epsilon > np.random.random():
            selected_action = np.random.random_integers(0, self.action_dim-1)
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            )
            selected_action = np.argmax(selected_action.detach().cpu().numpy())

        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.int64) -> Tuple[np.ndarray, np.float64, bool]:
        "Take an action and return environment response"
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]   
        reward = env_info.rewards[0]                   
        done = env_info.local_done[0]
    
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """ Update model by gradient descent"""
        samples = self.memory.sample_batch()
        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_episode: int, max_iteration: int=1000, plotting_interval: int=400):
        """  train the agent """
        self.is_test = False

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]

        update_cnt = 0
        epsilons = []
        losses = []
        avg_losses= []
        scores = []
        avg_scores = []

        for episode in range(num_episode):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0
            for iter in range(max_iteration):
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward
                if done:
                    break

                if len(self.memory) > self.batch_size:
                    loss = self.update_model()
                    losses.append(loss)
                    update_cnt += 1

            avg_losses.append(np.mean(losses))
            losses = []
            self.epsilon = max(
                self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
                ) * self.epsilon_decay
            )
            epsilons.append(self.epsilon)
            
            if update_cnt % self.target_update == 0:
                self._target_hard_update()
            scores.append(score)
            epsilons.append(self.epsilon)

            if episode >= 100:
                avg_scores.append(np.mean(scores[-100:]))
            self._plot(episode, scores, avg_scores, avg_losses, epsilons)
        torch.save(self.dqn.state_dict(), "model_weight/dqn.pt")



    def test(self):
        """ Test agent """
        self.is_test = True
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        state = env_info.vector_observations[0]
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()


    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float=0.99) -> torch.Tensor:
        """ Compute and return DQN loss"""
        gamma = self.gamma
        device = self.device
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).reshape(-1, 1).to(device)
        reward = torch.FloatTensor(samples["rews"]).reshape(-1, 1).to(device)
        done = torch.FloatTensor(samples["done"]).reshape(-1, 1).to(device)
        
        curr_q_value = self.dqn(state).gather(1, action)
            
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(device)
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss


    def _target_hard_update(self):
        """ update target network """
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
        self,
        episode :int,
        scores: List[float],
        avg_scores: List[float],
        losses: List[float],
        epsilons: List[float]
    ):
        """ Plot the training process"""
        plt.figure(figsize=(20, 5))
        plt.subplot(141)
        if len(avg_scores) > 0:
            plt.title("Average reward per 100 episodes. Score: %s" % (avg_scores[-1]))
        else:
            plt.title("Average reward over 100 episodes.")
        plt.plot([100 + i for i in range(len(avg_scores))], avg_scores)
        plt.subplot(142)
        plt.title("episode %s. Score: %s" % (episode, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(143)
        plt.title('Loss')
        plt.plot(losses)
        plt.subplot(144)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.savefig('plots/dqn_result.png')
