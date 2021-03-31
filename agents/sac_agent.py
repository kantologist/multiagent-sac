""" Packaged MASAC"""
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from unityagents import UnityEnvironment
from buffers.buffer import ReplayBuffer
from models.actor_critic import Actor, CriticQ, CriticV
from torch.nn.utils.clip_grad import clip_grad_norm_


class SacAgent:
    def __init__(
        self,
        env: UnityEnvironment,
        memory_size: int,
        batch_size:int,
        gamma: float=0.99,
        tau: float=5e-3,
        initial_random_steps: int= int(1e4),
        policy_update_fequency: int=2,
        ):
        
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        env_info = env.reset(train_mode=True)[self.brain_name]
        self.env = env
        self.action_size = self.brain.vector_action_space_size
        state = env_info.vector_observations[0]
        self.state_size = len(state)
        
        self.memory = ReplayBuffer(self.state_size, self.action_size, memory_size, batch_size)
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_frequecy = policy_update_fequency

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            )
        
        self.target_alpha = -np.prod((self.action_size,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = Actor(self.state_size, self.action_size).to(self.device)

        self.vf = CriticV(self.state_size).to(self.device)
        self.vf_target = CriticV(self.state_size).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        self.qf1 = CriticQ(self.state_size + self.action_size).to(self.device)
        self.qf2 = CriticQ(self.state_size + self.action_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=3e-4)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=3e-4)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=3e-4)

        self.transition = list()

        self.total_step = 0

        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = np.random.uniform(-1, 1, self.action_size)
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            )[0].detach().cpu().numpy()
        
        self.transition = [state, selected_action]

        return selected_action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:

        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]   
        reward = env_info.rewards[0]                   
        done = env_info.local_done[0]

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
        
        return next_state, reward, done
    
    def update_model(self) -> Tuple[torch.Tensor, ...]:
        device = self.device

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, self.action_size)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        new_action, log_prob = self.actor(state)


        alpha_loss = (
            -self.log_alpha.exp() * (log_prob + self.target_alpha).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        mask = 1 - done
        q1_pred = self.qf1(state, action)
        q2_pred = self.qf2(state, action)
        vf_target = self.vf_target(next_state)
        q_target = reward + self.gamma * vf_target * mask
        qf1_loss = F.mse_loss(q_target.detach(), q1_pred)
        qf2_loss = F.mse_loss(q_target.detach(), q2_pred)

        v_pred = self.vf(state)
        q_pred = torch.min(
            self.qf1(state, new_action), self.qf2(state, new_action)
        )
        v_target = q_pred - alpha * log_prob
        v_loss = F.mse_loss(v_pred, v_target.detach())

        if self.total_step % self.policy_update_frequecy == 0:
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)
        
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        qf_loss = qf1_loss + qf2_loss

        self.vf_optimizer.zero_grad()
        v_loss.backward()
        self.vf_optimizer.step()

        return actor_loss.data, qf_loss.data, v_loss.data, alpha_loss.data
    
    def train(self, num_episode: int, max_iteration: int=1000, plotting_interval: int=400):
        self.is_test = False

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]

        actor_losses, qf_losses, v_losses, alpha_losses = [], [], [], []
        scores = []
        avg_scores = []
        score = 0

        for episode in range(num_episode):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0

            for iter in range(max_iteration):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward
                if done:
                    break

                if (
                    len(self.memory) > self.batch_size
                and self.total_step > self.initial_random_steps):
                    loss = self.update_model()
                    actor_losses.append(loss[0])
                    qf_losses.append(loss[1])
                    v_losses.append(loss[2])
                    alpha_losses.append(loss[3])
            
            scores.append(score)
            if episode >= 100:
                avg_scores.append(np.mean(scores[-100:]))
            self._plot(episode, scores, avg_scores, actor_losses, qf_losses, v_losses, alpha_losses)
        torch.save(self.actor.state_dict(), "model_weight/actor.pt")
        torch.save(self.qf1.state_dict(), "model_weight/qf1.pt")
        torch.save(self.qf2.state_dict(), "model_weight/qf2.pt")
        torch.save(self.vf.state_dict(), "model_weight/vf.pt")
        
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
    
    def _target_soft_update(self):
        tau = self.tau

        for t_param, l_param in zip(
            self.vf_target.parameters(), self.vf.parameters()
        ):
            t_param.data.copy_( tau * l_param.data + (1.0 - tau) * t_param.data)

    def _plot(
        self,
        episode: int,
        scores: List[float],
        avg_scores: List[float],
        actor_losses: List[float],
        qf_losses: List[float],
        v_losses: List[float],
        alpha_losses: List[float]
        ):
        plt.figure(figsize=(20, 5))
        plt.subplot(121)
        if len(avg_scores) > 0:
            plt.title("Average reward per 100 episodes. Score: %s" % (avg_scores[-1]))
        else:
            plt.title("Average reward over 100 episodes.")
        plt.plot([100 + i for i in range(len(avg_scores))], avg_scores)
        plt.subplot(122)
        plt.title("episode %s. Score: %s" % (episode, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.savefig('plots/sac_result.png')
        plt.close()

        plt.figure(figsize=(20, 5))
        plt.subplot(141)
        plt.title('Actor Loss')
        plt.plot(actor_losses)
        plt.subplot(142)
        plt.title('qf loss')
        plt.plot(qf_losses)
        plt.subplot(143)
        plt.title('Vf Loss')
        plt.plot(v_losses)
        plt.subplot(144)
        plt.title('alpha loss')
        plt.plot(alpha_losses)
        plt.savefig('plots/sac_loss.png')
                




        