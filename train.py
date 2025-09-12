#!/usr/bin/env python3

import numpy as np
import torch
import argparse
import os

import replay_buffer
import TD3_OE
import rospy
import task_env
import os
from os import listdir
from os.path import isfile, join
import yaml
from yaml.loader import SafeLoader
import time
import ruamel.yaml
import datetime
import wandb


USE_WANDB = True

# Initialize WandB
if USE_WANDB:
	wandb.init(project="TD3_OE", name="default")


dir_path = os.path.dirname(os.path.realpath(__file__))

max_time_steps = 800 

def eval_policy(policy, env, eval_episodes=6):
	global max_time_steps
	print("---------------------------------------")
	print("Starting evaluation")
	print("---------------------------------------")
	avg_reward = 0.
	for _ in range(eval_episodes):
		vel, goal, state, done = env.reset()
		counter = 0
		while not done:
			counter += 1
			action = policy.select_action(state, vel, goal)
			vel, goal, state, reward, done = env.step(action)
			avg_reward += reward

			if counter > max_time_steps:
				done = True


	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


def main():

	global curriculum, curriculum_counter, max_time_steps 
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3_OE")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="default")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=5e3, type=int)# Time steps initial random policy is used 25e3 ### 50e3, 2e3, 8e3
	parser.add_argument("--eval_freq", default=16e3, type=int)       # How often (time steps) we evaluate - Before 5e3, now 3e4 ### 25e3, 2e3
	parser.add_argument("--max_timesteps", default=6e6, type=int)   # Max time steps to run environment. Between 15.000 to 150.000 episodes #1e7
	parser.add_argument("--max_time_per_episode", default=200, type=int)  # Max episodes to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise ### Before 0.1
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic - Before 256
	parser.add_argument("--discount", default=0.95)                 # Discount factor - before 0.99 -- 0.9
	parser.add_argument("--tau", default=0.005)                     # Target network update rate -- 0.005
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update ### Before 0.2
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise ### Before 0.5
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates ### Before 2 ############## CDRL 4 #######
	parser.add_argument("--save_model", default = True, action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default=False)                 # Model load file name, "" doesn't load, "default" uses file_name
	args, unknown = parser.parse_known_args()
	

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")


	env = task_env.PtdrlTaskEnv()
	episode_num, min_eval, time_step, ptr, size = get_env_params()
	print(f"episode_num: {episode_num}")

	# Set seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = 720 * 4 # 720 Lidar size. 4 Frames
	action_dim = 2 # Linear and angular velocity
	max_action = 2.0 

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3_OE.TD3(**kwargs)
	

	if args.load_model:
		policy_file = file_name
		policy.load(f"{dir_path}/models_TD3_cdrl/{policy_file}", "current")
		print("Loading Network!!!")

	replay_buffer = replay_buffer.ReplayBufferLidar(state_dim, action_dim)

	# Start training
	vel, goal, state, done = env.reset()
	
	episode_reward = 0
	episode_timesteps = 0

	for t in range(time_step, int(args.max_timesteps)):
		episode_timesteps += 1
		dwa_action = env.get_cmd_vel_dwa()
		# Select action randomly or according to policy
		if t < args.start_timesteps:
			# Random around move_base values
			action = env.randomize_action(dwa_action)
		else:
			action = (
				policy.select_action(np.array(state), vel, goal)
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

			action = (
				policy.select_action(np.array(state), vel, goal))
			
		next_vel, next_goal, next_state, reward, done = env.step_cdrl_short_lidar(action)

		if episode_timesteps > max_time_steps:
			reward = -1
			done = 1

		done_bool = float(done) if episode_timesteps < args.max_time_per_episode else 0 

		# Store data in replay buffer
		replay_buffer.add(vel, goal, state, action, dwa_action, next_vel, next_goal, next_state, reward, done_bool)

		vel = next_vel
		goal = next_goal
		state = next_state
		episode_reward += reward

		# Train agent
		if t >= args.start_timesteps:
			critic_loss, actor_loss = policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"t:{t}")
			print(f"episode_num: {episode_num}")
			print(f"episode_timesteps: {episode_timesteps}")
			print(f"episode_reward: {episode_reward}")
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment

			if USE_WANDB:
				wandb.log({"episode_reward": episode_reward}, step=episode_num)
		
			vel, goal, state, done = env.reset()
			print("start Reset")
			time.sleep(2.0) # Wait for reset to finish
			print("Finsih reset")
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 


		# Evaluate episode
		if args.save_model:
			if (t + 1) % args.eval_freq == 0:
				policy.save(f"{dir_path}/models/{file_name}", "current")
				eval = eval_policy(policy, env)
				
				if (eval > min_eval) and (eval != 0): 
					policy.save(f"{dir_path}/models/{file_name}", "best_" + str(int(t/1e3)) + str(int(eval)))
					min_eval = eval

if __name__ == '__main__':
    rospy.init_node('init_train')
    main()
