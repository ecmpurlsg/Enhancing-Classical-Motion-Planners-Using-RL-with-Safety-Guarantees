import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conv_gru import ConvGRUActor, ConvGRUCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self._actor = ConvGRUActor()
        self.max_action = max_action
        
    def forward(self, state, vel, goal):
        return self.max_action * torch.tanh(self._actor(state, vel, goal))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self._q1 = ConvGRUCritic()
        self._q2 = ConvGRUCritic()

    def forward(self, state, vel, goal, action):
        q1 = self._q1(state, vel, goal, action)
        q2 = self._q2(state, vel, goal, action)
        return q1, q2

    def Q1(self, state, vel, goal, action):
        return self._q1(state, vel, goal, action)

class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.95,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2 
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.g_max  = 10.0    
        self.beta_gp = 1e-2
        self.alpha = 2.5 
        self.epsilon_max = 0.5 
        self.lambda_0 = 2 * self.epsilon_max / self.g_max

        self.total_it = 0

    def select_action(self, state, vel, goal):
        vel = torch.cuda.FloatTensor(vel).to(device).unsqueeze(0)
        goal = torch.cuda.FloatTensor(goal).to(device).unsqueeze(0)
        state = torch.cuda.FloatTensor(state).to(device).unsqueeze(0)
        return self.actor(state, vel, goal).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        vel, goal, state, action, dwa_action, next_vel, next_goal, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # ────────── 1. CRITIC UPDATE ───────────────────────────────────────────
        
        # Action-gradient tracking for both critics (Equation 2)
        a_lip = action.detach().clone().requires_grad_(True)
        q1_tmp, q2_tmp = self.critic(state, vel, goal, a_lip)
        
        # Compute individual gradients
        grad_a1 = torch.autograd.grad(q1_tmp.sum(), a_lip, create_graph=True)[0]
        grad_a2 = torch.autograd.grad(q2_tmp.sum(), a_lip, create_graph=True)[0]
        
        # Joint Lipschitz penalty: 1/2 * (||grad_Q1|| + ||grad_Q2||)
        avg_grad_norm = 0.5 * (grad_a1.norm(dim=1) + grad_a2.norm(dim=1))
        gp = F.relu(avg_grad_norm - self.g_max).pow(2).mean()

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, next_vel, next_goal) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_vel, next_goal, next_action)
            target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(state, vel, goal, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + self.beta_gp * gp

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # ────────── 2. DELAYED ACTOR UPDATE ─────────────────────────────────────
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state, vel, goal)
            Q1_pi = self.critic.Q1(state, vel, goal, pi)

            # Adaptive Weight Lambda calculation
            lambda_adapt = self.alpha / Q1_pi.abs().mean().detach()
            lambda_clip = torch.minimum(lambda_adapt, torch.tensor(self.lambda_0, device=Q1_pi.device))

            # RL loss (scaled by lambda) + BC loss (unscaled)
            # F.mse_loss computes the square term from Equation 1
            actor_loss = -lambda_clip * Q1_pi.mean() + F.mse_loss(pi, dwa_action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # Target Network Updates
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if actor_loss is not None:
            return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()
        return critic_loss.cpu().data.numpy(), -1

    def save(self, filename, current_best):
        torch.save(self.critic.state_dict(), filename + "_critic_" + current_best)
        torch.save(self.actor.state_dict(), filename + "_actor_" + current_best)

    def load(self, filename, current_best):
        self.critic.load_state_dict(torch.load(filename + "_critic_" + current_best))
        self.actor.load_state_dict(torch.load(filename + "_actor_" + current_best))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

		
