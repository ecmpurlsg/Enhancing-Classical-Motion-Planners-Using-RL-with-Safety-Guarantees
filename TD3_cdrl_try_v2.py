import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cdrl_o3_v1 import ConvGRUActor, ConvGRUCritic

from PIL import Image
import matplotlib.pyplot as plt

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
		sa = state
		v = vel
		g = goal


		q1 = self._q1(sa, v, g, action)

		q2 = self._q2(sa, v, g, action)

		return q1, q2


	def Q1(self, state, vel, goal, action):
		sa = state
		v = vel
		g = goal

		q1 = self._q1(sa, v, g,  action)

		return q1



class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.95, #0.9
		tau=0.005,
		policy_noise=0.2, # 0.2
		noise_clip=0.5, # 0.5
		policy_freq=2 # 4
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4) # Before self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)# Before self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.g_max  = 10.0    # 5 # Lipschitz cap  ‖∇a Q‖ ≤ gmax
		self.beta_gp = 1e-2
		self.alpha = 2.5 # 5.0
		self.epsilon_max = 0.5 # 2 
		self.lambda_0 = 2 * self.epsilon_max / self.g_max

		self.total_it = 0


	def select_action(self, state, vel, goal):

		vel = torch.cuda.FloatTensor(vel).to(device)
		vel = vel.unsqueeze(0)
		goal = torch.cuda.FloatTensor(goal).to(device)
		goal = goal.unsqueeze(0)
		state = torch.cuda.FloatTensor(state).to(device)
		state = state.unsqueeze(0)
		return self.actor(state, vel, goal).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256): #256 !!!!!!!!!!!!!
		# 1. Compute pi with gradient tracking

		self.total_it += 1

		# Sample replay buffer 
		vel, goal, state, action, dwa_action, next_vel, next_goal, next_state, reward, not_done = replay_buffer.sample(batch_size)
		a_lip = action.detach().clone().requires_grad_(True)   # enable grads on actions
		q_tmp = self.critic.Q1(state, vel, goal, a_lip).sum()  # forward pass that uses a_lip

		grad_a = torch.autograd.grad(q_tmp, a_lip, create_graph=True)[0]  # (B, act_dim)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state, next_vel, next_goal) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_vel, next_goal, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, vel, goal, action)

		if self.total_it % 100 == 0:
			print(f"Current Q1 {current_Q1[0]}")
			print("!!!!!!!!!!!!!!!!!!!")
			print(f"Current Q2 {current_Q2[0]}")
			print("!!!!!!!!!!!!!!!!!!!")
			print(f"Current Target {target_Q[0]}")
			print("!!!!!!!!!!!!!!!!!!!")

		# ────────── 1. CRITIC update ───────────────────────────────────────────
		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# -------- Gradient-penalty part (enforces ‖∇a Q‖ ≤ gmax) --------
		grad_norm = grad_a.norm(dim=1)                        # (B,)
		# hinge penalty: only active where ∥∇Q∥ > gmax
		gp = F.relu(grad_norm - self.g_max).pow(2).mean()
		critic_loss = critic_loss + self.beta_gp * gp
		# ----------------------------------------------------------------

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
		self.critic_optimizer.step()

		actor_loss = None
		# Delayed policy updates
		# Actor update
		if self.total_it % self.policy_freq == 0:

			pi = self.actor(state, vel, goal)                 # single forward pass
			Q1_pi = self.critic.Q1(state, vel, goal, pi)

			lambda_adapt = self.alpha / Q1_pi.abs().mean().detach()
			lambda_clip  = torch.minimum(lambda_adapt,
										torch.tensor(self.lambda_0,
													device=Q1_pi.device))

			# BC term only where linear part of DWA action is non-zero
			mask = (dwa_action[:, 0] != 0.0)
			bc_loss = F.mse_loss(pi[mask], dwa_action[mask])
			bc_loss = F.mse_loss(pi, dwa_action)

			actor_loss = -lambda_clip * Q1_pi.mean() + bc_loss

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
			self.actor_optimizer.step()

			if self.total_it % 200 == 0:
				print(f"bc_loss: {bc_loss:.4f}")
				print(f"‖∇aQ‖_max {grad_norm.max():.2f} ")
				print(f"lambda_clipped: {lambda_clip:.4f}, lambda_adaptive: {lambda_adapt:.4f} ")
				print(f"pi: {pi[0:10]}")
				print(f"dwa: {dwa_action[0:10]}")


			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		if actor_loss is not None:
			return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()
		return critic_loss.cpu().data.numpy(), -1


	def save(self, filename, current_best):
		torch.save(self.critic.state_dict(), filename + "_critic_" + current_best)
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer_" + current_best)
		
		torch.save(self.actor.state_dict(), filename + "_actor_" + current_best)
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer_" + current_best)


	def load(self, filename, current_best):
		self.critic.load_state_dict(torch.load(filename + "_critic_" + current_best))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer_" + current_best))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor_" + current_best))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer_" + current_best))
		self.actor_target = copy.deepcopy(self.actor)

	def load_dpo(self, filename, filename_actor):
		self.critic.load_state_dict(torch.load(filename + "_critic_current"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer_current"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename_actor))
		self.setup_model_for_inference(self.actor)
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer_current"))
		self.actor_target = copy.deepcopy(self.actor)

	def setup_model_for_inference(self, model):
		# Put the model in evaluation mode
		model.eval()

		# Disable dropout layers
		def disable_dropout(m):
			if type(m) == torch.nn.Dropout:
				m.train(False)
		model.apply(disable_dropout)

		# Set batch normalization layers to evaluation mode
		def set_bn_eval_mode(m):
			if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
				m.eval()
		model.apply(set_bn_eval_mode)

		# Freeze gradients for all parameters
		for param in model.parameters():
			param.requires_grad = False
		