import numpy as np
import torch

class RunningNorm:
	def __init__(self, shape, epsilon=1e-8):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.mean = torch.zeros(shape).to(self.device)
		self.var = torch.ones(shape).to(self.device)
		self.count = epsilon

	def update(self, x):
		batch_mean = x.mean(dim=0)
		batch_var = x.var(dim=0, unbiased=False)
		batch_count = x.size(0)

		delta = batch_mean - self.mean
		total_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / total_count
		m_a = self.var * self.count
		m_b = batch_var * batch_count
		M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
		new_var = M2 / total_count

		self.mean, self.var, self.count = new_mean, new_var, total_count

	def normalize(self, x):
		return (x - self.mean) / (self.var.sqrt() + 1e-8)

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e5)): # - Before 1e6
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.dwa_action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, dwa_action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.dwa_action[self.ptr] = dwa_action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.dwa_action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
	def save(self, dir_path):
		np.save(dir_path + "/models_TD3_cdrl/buffer_state.npy", self.state)
		np.save(dir_path + "/models_TD3_cdrl/buffer_action.npy", self.action)
		np.save(dir_path + "/models_TD3_cdrl/buffer_dwa_action.npy", self.dwa_action)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_state.npy", self.next_state)
		np.save(dir_path + "/models_TD3_cdrl/buffer_reward.npy", self.reward)
		np.save(dir_path + "/models_TD3_cdrl/buffer_not_done.npy", self.not_done)

	def load(self, dir_path):
		self.state = np.load(dir_path + "/models_TD3_cdrl/buffer_state.npy")
		self.action = np.load(dir_path + "/models_TD3_cdrl/buffer_action.npy")
		self.dwa_action = np.load(dir_path + "/models_TD3_cdrl/buffer_dwa_action.npy")
		self.next_state = np.load(dir_path + "/models_TD3_cdrl/buffer_next_state.npy")
		self.reward = np.load(dir_path + "/models_TD3_cdrl/buffer_reward.npy")
		self.not_done = np.load(dir_path + "/models_TD3_cdrl/buffer_not_done.npy")


class ReplayBufferGoal(object):
	def __init__(self, state_dim, action_dim, max_size=int(2e5)): # - Before 1e6
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.vel = np.zeros((max_size, 2))
		self.goal = np.zeros((max_size, 2))
		self.state = np.zeros((max_size, state_dim, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.dwa_action = np.zeros((max_size, action_dim))
		self.next_vel = np.zeros((max_size, 2))
		self.next_goal = np.zeros((max_size, 2))
		self.next_state = np.zeros((max_size, state_dim, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, vel, goal, state, action, dwa_action, next_vel, next_goal, next_state, reward, done):
		self.vel[self.ptr] = vel
		self.goal[self.ptr] = goal
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.dwa_action[self.ptr] = dwa_action
		self.next_vel[self.ptr] = next_vel
		self.next_goal[self.ptr] = next_goal
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.vel[ind]).to(self.device),
			torch.FloatTensor(self.goal[ind]).to(self.device),
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.dwa_action[ind]).to(self.device),
			torch.FloatTensor(self.next_vel[ind]).to(self.device),
			torch.FloatTensor(self.next_goal[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
	def save(self, dir_path):
		np.save(dir_path + "/models_TD3_cdrl/buffer_vel.npy", self.vel)
		np.save(dir_path + "/models_TD3_cdrl/buffer_goal.npy", self.goal)
		np.save(dir_path + "/models_TD3_cdrl/buffer_state.npy", self.state)
		np.save(dir_path + "/models_TD3_cdrl/buffer_action.npy", self.action)
		np.save(dir_path + "/models_TD3_cdrl/buffer_dwa_action.npy", self.dwa_action)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_vel.npy", self.next_vel)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_goal.npy", self.next_goal)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_state.npy", self.next_state)
		np.save(dir_path + "/models_TD3_cdrl/buffer_reward.npy", self.reward)
		np.save(dir_path + "/models_TD3_cdrl/buffer_not_done.npy", self.not_done)

	def load(self, dir_path):
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_vel.npy")
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_goal.npy")
		self.state = np.load(dir_path + "/models_TD3_cdrl/buffer_state.npy")
		self.action = np.load(dir_path + "/models_TD3_cdrl/buffer_action.npy")
		self.dwa_action = np.load(dir_path + "/models_TD3_cdrl/buffer_dwa_action.npy")
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_next_vel.npy")
		self.next_goal = np.load(dir_path + "/models_TD3_cdrl/buffer_next_goal.npy")
		self.next_state = np.load(dir_path + "/models_TD3_cdrl/buffer_next_state.npy")
		self.reward = np.load(dir_path + "/models_TD3_cdrl/buffer_reward.npy")
		self.not_done = np.load(dir_path + "/models_TD3_cdrl/buffer_not_done.npy")


class ReplayBufferLidar(object):
	def __init__(self, state_dim, action_dim, max_size=int(2e5)): # - Before 1e6
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.vel = np.zeros((max_size, 2))
		self.goal = np.zeros((max_size, 2))
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.dwa_action = np.zeros((max_size, 2))
		self.next_vel = np.zeros((max_size, 2))
		self.next_goal = np.zeros((max_size, 2))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.vel_normalizer = RunningNorm(2)
		self.goal_normalizer = RunningNorm(2)
		self.state_normalizer = RunningNorm(state_dim)


	def add(self, vel, goal, state, action, dwa_action, next_vel, next_goal, next_state, reward, done):
		self.vel[self.ptr] = vel
		self.goal[self.ptr] = goal
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.dwa_action[self.ptr] = dwa_action
		self.next_vel[self.ptr] = next_vel
		self.next_goal[self.ptr] = next_goal
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size, normalize=False):
		ind = np.random.randint(0, self.size, size=batch_size)

		vel         = torch.FloatTensor(self.vel[ind]).to(self.device)
		goal        = torch.FloatTensor(self.goal[ind]).to(self.device)
		state       = torch.FloatTensor(self.state[ind]).to(self.device)
		action      = torch.FloatTensor(self.action[ind]).to(self.device)
		dwa_action  = torch.FloatTensor(self.dwa_action[ind]).to(self.device)
		next_vel    = torch.FloatTensor(self.next_vel[ind]).to(self.device)
		next_goal   = torch.FloatTensor(self.next_goal[ind]).to(self.device)
		next_state  = torch.FloatTensor(self.next_state[ind]).to(self.device)
		reward      = torch.FloatTensor(self.reward[ind]).to(self.device)
		not_done    = torch.FloatTensor(self.not_done[ind]).to(self.device)

		if normalize:
			self.state_normalizer.update(state)
			state = self.state_normalizer.normalize(state)
			next_state = self.state_normalizer.normalize(next_state)

			self.vel_normalizer.update(vel)
			vel = self.vel_normalizer.normalize(vel)
			next_vel = self.vel_normalizer.normalize(vel)

			self.goal_normalizer.update(goal)
			goal = self.goal_normalizer.normalize(goal)
			next_goal = self.goal_normalizer.normalize(goal)

		return (
			vel,
			goal,
			state,
			action,
			dwa_action,
			next_vel,
			next_goal,
			next_state,
			reward,
			not_done
		)
	
	def save(self, dir_path):
		np.save(dir_path + "/models_TD3_cdrl/buffer_vel.npy", self.vel)
		np.save(dir_path + "/models_TD3_cdrl/buffer_goal.npy", self.goal)
		np.save(dir_path + "/models_TD3_cdrl/buffer_state.npy", self.state)
		np.save(dir_path + "/models_TD3_cdrl/buffer_action.npy", self.action)
		np.save(dir_path + "/models_TD3_cdrl/buffer_dwa_action.npy", self.dwa_action)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_vel.npy", self.next_vel)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_goal.npy", self.next_goal)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_state.npy", self.next_state)
		np.save(dir_path + "/models_TD3_cdrl/buffer_reward.npy", self.reward)
		np.save(dir_path + "/models_TD3_cdrl/buffer_not_done.npy", self.not_done)

	def load(self, dir_path):
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_vel.npy")
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_goal.npy")
		self.state = np.load(dir_path + "/models_TD3_cdrl/buffer_state.npy")
		self.action = np.load(dir_path + "/models_TD3_cdrl/buffer_action.npy")
		self.dwa_action = np.load(dir_path + "/models_TD3_cdrl/buffer_dwa_action.npy")
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_next_vel.npy")
		self.next_goal = np.load(dir_path + "/models_TD3_cdrl/buffer_next_goal.npy")
		self.next_state = np.load(dir_path + "/models_TD3_cdrl/buffer_next_state.npy")
		self.reward = np.load(dir_path + "/models_TD3_cdrl/buffer_reward.npy")
		self.not_done = np.load(dir_path + "/models_TD3_cdrl/buffer_not_done.npy")



class ReplayBufferLidarResidual(object):
	def __init__(self, state_dim, action_dim, max_size=int(2e5)): # - Before 1e6
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.vel = np.zeros((max_size, 2))
		self.goal = np.zeros((max_size, 2))
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.dwa_action = np.zeros((max_size, action_dim))
		self.next_vel = np.zeros((max_size, 2))
		self.next_goal = np.zeros((max_size, 2))
		self.next_state = np.zeros((max_size, state_dim))
		self.next_dwa_action = np.zeros((max_size, action_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.vel_normalizer = RunningNorm(2)
		self.goal_normalizer = RunningNorm(2)
		self.state_normalizer = RunningNorm(state_dim)


	def add(self, vel, goal, state, action, dwa_action, next_vel, next_goal, next_state, next_dwa_action, reward, done):
		self.vel[self.ptr] = vel
		self.goal[self.ptr] = goal
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.dwa_action[self.ptr] = dwa_action
		self.next_vel[self.ptr] = next_vel
		self.next_goal[self.ptr] = next_goal
		self.next_state[self.ptr] = next_state
		self.next_dwa_action[self.ptr] = next_dwa_action
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size, normalize=False):
		ind = np.random.randint(0, self.size, size=batch_size)

		vel         = torch.FloatTensor(self.vel[ind]).to(self.device)
		goal        = torch.FloatTensor(self.goal[ind]).to(self.device)
		state       = torch.FloatTensor(self.state[ind]).to(self.device)
		action      = torch.FloatTensor(self.action[ind]).to(self.device)
		dwa_action  = torch.FloatTensor(self.dwa_action[ind]).to(self.device)
		next_vel    = torch.FloatTensor(self.next_vel[ind]).to(self.device)
		next_goal   = torch.FloatTensor(self.next_goal[ind]).to(self.device)
		next_state  = torch.FloatTensor(self.next_state[ind]).to(self.device)
		next_dwa_action  = torch.FloatTensor(self.next_dwa_action[ind]).to(self.device)
		reward      = torch.FloatTensor(self.reward[ind]).to(self.device)
		not_done    = torch.FloatTensor(self.not_done[ind]).to(self.device)

		if normalize:
			self.state_normalizer.update(state)
			state = self.state_normalizer.normalize(state)
			next_state = self.state_normalizer.normalize(next_state)

			self.vel_normalizer.update(vel)
			vel = self.vel_normalizer.normalize(vel)
			next_vel = self.vel_normalizer.normalize(vel)

			self.goal_normalizer.update(goal)
			goal = self.goal_normalizer.normalize(goal)
			next_goal = self.goal_normalizer.normalize(goal)

		return (
			vel,
			goal,
			state,
			action,
			dwa_action,
			next_vel,
			next_goal,
			next_state,
			next_dwa_action,
			reward,
			not_done
		)
	
	def save(self, dir_path):
		np.save(dir_path + "/models_TD3_cdrl/buffer_vel.npy", self.vel)
		np.save(dir_path + "/models_TD3_cdrl/buffer_goal.npy", self.goal)
		np.save(dir_path + "/models_TD3_cdrl/buffer_state.npy", self.state)
		np.save(dir_path + "/models_TD3_cdrl/buffer_action.npy", self.action)
		np.save(dir_path + "/models_TD3_cdrl/buffer_dwa_action.npy", self.dwa_action)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_vel.npy", self.next_vel)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_goal.npy", self.next_goal)
		np.save(dir_path + "/models_TD3_cdrl/buffer_next_state.npy", self.next_state)
		np.save(dir_path + "/models_TD3_cdrl/buffer_reward.npy", self.reward)
		np.save(dir_path + "/models_TD3_cdrl/buffer_not_done.npy", self.not_done)

	def load(self, dir_path):
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_vel.npy")
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_goal.npy")
		self.state = np.load(dir_path + "/models_TD3_cdrl/buffer_state.npy")
		self.action = np.load(dir_path + "/models_TD3_cdrl/buffer_action.npy")
		self.dwa_action = np.load(dir_path + "/models_TD3_cdrl/buffer_dwa_action.npy")
		self.goal = np.load(dir_path + "/models_TD3_cdrl/buffer_next_vel.npy")
		self.next_goal = np.load(dir_path + "/models_TD3_cdrl/buffer_next_goal.npy")
		self.next_state = np.load(dir_path + "/models_TD3_cdrl/buffer_next_state.npy")
		self.reward = np.load(dir_path + "/models_TD3_cdrl/buffer_reward.npy")
		self.not_done = np.load(dir_path + "/models_TD3_cdrl/buffer_not_done.npy")
