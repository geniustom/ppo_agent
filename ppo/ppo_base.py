from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
import gym
import time
import datetime
import os, os.path
import warnings
import pickle
from shutil import copyfile

# python – keras：如何保存模型并继续培训？ https://codeday.me/bug/20180921/257413.html
# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
import numpy as np

import gym
import collections
from statistics import mean, stdev

from tensorflow.keras.utils import plot_model
from scipy.stats import entropy

import numba as nb
from tensorboardX import SummaryWriter
import ppo.ai_lib.model as am


class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

class PPOAgentBase(ABC):
	def __init__(self, env, experiment_name, 
							actor_model=None, 
							critic_model=None,
							LEARNING_RATE=2.5e-4, 
							LEARNING_RATE_TARGET=0, 
							LOSS_CLIPPING=0.2,
							EARLY_STOPPING_KL_AMOUNT=0.01, 
							ENTROPY_LOSS=0.01, 
							GAMMA=0.99, 
							BATCH_SIZE=32, 
							BUFFER_SIZE=256, 
							EPOCHS=4, 
							TRAINING_STEP_LENGTH=1e4,
							log_frequency=1000,
							network_style=None):

		self.ENV_NAME = env.unwrapped.spec.id

		print(f"Got in the {self.ENV_NAME} env.")
		self.env = env
		self.experiment_name = experiment_name
	
		print(f"The action space is: {self.env.unwrapped.action_space}")
		print(f"The observation space is: {self.env.unwrapped.observation_space}")

		# TODO: Add support for continious action space.
		self.action_shape = self.env.unwrapped.action_space.n
		self.observation_shape = self.env.unwrapped.observation_space.shape
		self.actor_model=actor_model
		self.critic_model=critic_model
		self.network_style = network_style

		# Algorithm Hyperparameters
		self.LEARNING_RATE = LEARNING_RATE
		self.LEARNING_RATE_TARGET = LEARNING_RATE_TARGET
		self.LOSS_CLIPPING = LOSS_CLIPPING
		self.EARLY_STOPPING_KL_AMOUNT = EARLY_STOPPING_KL_AMOUNT
		self.ENTROPY_LOSS = ENTROPY_LOSS
		self.GAMMA = GAMMA
		self.BATCH_SIZE = BATCH_SIZE
		self.BUFFER_SIZE = BUFFER_SIZE
		self.EPOCHS = EPOCHS
		self.TRAINING_STEP_LENGTH = TRAINING_STEP_LENGTH
	  
		# Logging Paths.
		self.log_frequency = log_frequency
		self.result_dir_path = self.get_summary_path(experiment_name)
		self.writer = SummaryWriter(self.result_dir_path)
		self.log_file = open(f"{self.result_dir_path}/data.txt", "w")

		self.last_fifty_epsiodes = collections.deque(50*[0], 50)
		self.last_fifty_epsiodes_lengths = collections.deque(50*[0], 50)
		self.last_fifty_deltas = collections.deque(50*[0], 50)
		self.last_hundred_kl_exits = collections.deque(100*[0], 100)
		self.max_best_mean = -np.inf
		self.best_reward = -np.inf

		# Models
		self.actor, self.critic = self.build_actor_critic()
		self.ACTModel=am.GModel(model=self.actor,path="./"+self.ENV_NAME+"_actor.h5")
		self.CRTModel=am.GModel(model=self.critic,path="./"+self.ENV_NAME+"_critic.h5")

		# These are used when you are prediction the action.
		self.DUMMY_ACTION = np.zeros((1, self.action_shape))
		self.DUMMY_VALUE = np.zeros((1, 1))

		# Random Baseline information
		self.max_entropy = entropy([1.0] * self.action_shape)
		self.random_baseline_mean_std = self.get_lucky_random_baseline(100000)

		# Bookkeeping variables.
		self.episode_rewards = []
		self.gradient_steps = 0
		self.episode = 0
		self.steps = 0
		self.last_kl_divergence = {"value": 0}
		self.actor_loss, self.critic_loss = None, None
		
		self.observation = self.env.reset()
		self.premature_ending = False


	def get_lucky_random_baseline(self, max_steps, reset=False):
		env_path = os.path.join("Results", self.ENV_NAME)
		pickle_path = os.path.join(env_path, "random_baseline.pickle")
		if not reset and os.path.exists(pickle_path):
			with open(pickle_path, 'rb') as f:
				return pickle.load(f)

		episode_reward = []
		cumulative_rewards = []
		
		print("Starting to create the random baseline.")
		self.env.reset()
		for step in range(max_steps):
			action = self.env.action_space.sample() 
			_, reward, done, _ = self.env.step(action)
			episode_reward.append(reward)
			if done:
				ep_rew_sum = sum(episode_reward)
				#print(f"Step {step} out of {max_steps} with reward {ep_rew_sum}.")
				cumulative_rewards.append(ep_rew_sum)
				episode_reward = []
				self.env.reset()

		cumulative_rewards = np.array(cumulative_rewards)
		random_results = (cumulative_rewards.mean(), cumulative_rewards.std()) 
		with open(pickle_path, 'wb') as f:
			pickle.dump(random_results, f)
		return random_results

	def get_summary_path(self, experiment_name): 

		# Todo: Possibly move this out to a support utils file
		date = datetime.datetime.now()
		day = date.day
		month = date.month
		
		result_dir_path = "Results"
		if not os.path.exists(result_dir_path):
			os.makedirs(result_dir_path)
		
		result_dir_path = os.path.join(result_dir_path, f"{self.ENV_NAME}")
		if not os.path.exists(result_dir_path):
			os.makedirs(result_dir_path)

		result_dir_path = os.path.join(result_dir_path, f"{month}_{day}")
		if not os.path.exists(result_dir_path):
			os.makedirs(result_dir_path)

		# Todo: Make the folder backfoundable.
		count = len([name for name in os.listdir(result_dir_path)])
		result_dir_path = os.path.join(result_dir_path, f"{experiment_name}")
		if not os.path.exists(result_dir_path):
			os.makedirs(result_dir_path)

		count = len([name for name in os.listdir(result_dir_path)])
		result_dir_path = os.path.join(result_dir_path, f"run_{count}")
		if not os.path.exists(result_dir_path):
			os.makedirs(result_dir_path)

		# self.LEARNING_RATE = LEARNING_RATE
		# self.LEARNING_RATE_TARGET = LEARNING_RATE_TARGET
		# self.LOSS_CLIPPING = LOSS_CLIPPING
		# self.EARLY_STOPPING_KL_AMOUNT = EARLY_STOPPING_KL_AMOUNT
		# self.ENTROPY_LOSS = ENTROPY_LOSS
		# self.GAMMA = GAMMA
		# self.BATCH_SIZE = BATCH_SIZE
		# self.BUFFER_SIZE = BUFFER_SIZE
		# self.EPOCHS = EPOCHS
		# self.TRAINING_STEP_LENGTH = TRAINING_STEP_LENGTH

		with open(os.path.join(f"{result_dir_path}", "Hyperparameters.txt"), "w") as readme:
			print("\n########################################################################", file=readme)
			print(f"#\n# The experiment {self.experiment_name} on the {self.ENV_NAME}.", file=readme)
			print(f"# The full environment and wrappers are: {self.env}.", file=readme)

			print(f"# The LEARNING_RATE: {self.LEARNING_RATE}.",file=readme)
			print(f"# The LEARNING_RATE_TARGET: {self.LEARNING_RATE_TARGET}.",file=readme)
			print(f"# The LOSS_CLIPPING: {self.LOSS_CLIPPING}.",file=readme)
			print(f"# The EARLY_STOPPING_KL_AMOUNT: {self.EARLY_STOPPING_KL_AMOUNT}.",file=readme)
			print(f"# The ENTROPY_LOSS: {self.ENTROPY_LOSS}.",file=readme)
			print(f"# The GAMMA: {self.GAMMA}.",file=readme)
			print(f"# The BATCH_SIZE: {self.BATCH_SIZE}.",file=readme)
			print(f"# The EPOCHS: {self.EPOCHS}.",file=readme)
			print(f"# The TRAINING_STEP_LENGTH: {self.TRAINING_STEP_LENGTH}.",file=readme)
			print(f"# The network_style: {self.network_style}.",file=readme)
		
		copyfile("ppo/ppo_variants.py", 
				 os.path.join(f"{result_dir_path}", "ppo_variants.py"))

		return result_dir_path
		
	def get_new_learning_rate(self):
		def lerp(p1, p2, perc):
			return p1 * (1 - perc) + p2 * perc

		if self.LEARNING_RATE_TARGET == None:
			return self.LEARNING_RATE
	
		percentage = self.steps / self.TRAINING_STEP_LENGTH
		new_lr = lerp(self.LEARNING_RATE, self.LEARNING_RATE_TARGET, percentage)
		return new_lr
	
	@abstractmethod
	def build_actor_critic(self):
		pass

	@abstractmethod
	def learn(self):
		pass

	def reset_env(self):
		self.episode += 1
		self.observation = self.env.reset()

	def get_action(self):
		probability_distribution = self.ACTModel.model.predict([np.expand_dims(self.observation, axis=0), self.DUMMY_VALUE, self.DUMMY_ACTION]).squeeze()
		action = np.random.choice(self.action_shape, p=np.nan_to_num(probability_distribution))

		action_matrix = np.zeros(self.action_shape)
		action_matrix[action] = 1
		return action, action_matrix, probability_distribution

	def log_information(self, observations, predictions):
		self.last_fifty_epsiodes.appendleft(sum(self.episode_rewards))
		self.last_fifty_epsiodes_lengths.appendleft(len(self.episode_rewards))
		if self.last_kl_divergence['value'] > self.EARLY_STOPPING_KL_AMOUNT:
			self.last_hundred_kl_exits.appendleft(1)
		else: 
			self.last_hundred_kl_exits.appendleft(0)

		if self.premature_ending:
			self.episode_rewards = self.episode_rewards[:-1]
			
		current_mean = mean(self.last_fifty_epsiodes)
		if self.max_best_mean < current_mean and self.episode>=50:
			self.max_best_mean = current_mean
			self.best_reward=self.max_best_mean
			self.actor.save("./"+self.ENV_NAME+"_actor.h5")
			self.critic.save("./"+self.ENV_NAME+"_critic.h5")
			print(f"The best model has been saved with the new {current_mean} reward.")

		episode_prediction = self.CRTModel.model.predict(np.expand_dims(observations[0], axis=0)).squeeze()
		episode_reward_sum = np.array(self.episode_rewards).sum()
		value_delta = episode_reward_sum - episode_prediction
		entropy_amount = entropy(np.transpose(predictions)).mean() 
		
		self.last_fifty_deltas.appendleft(value_delta)
		
		print()
		print(f"* ----------- Episode {self.episode} of {self.experiment_name} ----------- ")
		print(f"* Mean 50 Reward : {mean(self.last_fifty_epsiodes):.2f} +- {stdev(self.last_fifty_epsiodes):.2f}")
		print(f"* Random Agent   : {self.random_baseline_mean_std[0]:.2f} +- {self.random_baseline_mean_std[1]:.2f}")
		print(f"* Episode Reward : {episode_reward_sum:.2f}")
		print(f"* Mean 50 Ep Len : {mean(self.last_fifty_epsiodes_lengths):.2f} +- {stdev(self.last_fifty_epsiodes_lengths):.2f}")
		print(f"*")
		print(f"* Entropy Ratio  : {entropy_amount / self.max_entropy * 100:.2f} %")
		print(f"* Value Delta    : {value_delta:.2f}")
		print(f"* Mean 50 Deltas : {mean(self.last_fifty_deltas):.2f} +- {stdev(self.last_fifty_deltas):.2f}")
		print(f"* KL Divergence  : {self.last_kl_divergence['value']:.4f}")
		print(f"* KL Early Stops : {sum(self.last_hundred_kl_exits)} / 100")
		print(f"*")
		print(f"* Current LR     : {self.get_new_learning_rate():.3e}")
		print(f"* Gradient Steps : {self.gradient_steps}")
		print(f"* Steps          : {self.steps}")
		print(f"* Completion     : {self.steps / self.TRAINING_STEP_LENGTH * 100:.2f} %")
		print(f"* ----------------------------------- ") 
		
#		if self.actor_loss and self.actor_loss:
#			self.writer.add_scalar('Actor loss', self.actor_loss.history['loss'][-1], self.steps)
#			self.writer.add_scalar('Critic loss', self.critic_loss.history['loss'][-1], self.steps)

		self.writer.add_scalar("Entropy", entropy_amount, self.steps)
		self.writer.add_scalar("Value_delta", value_delta, self.steps)
		self.writer.add_scalar('Episode Reward', np.array(self.episode_rewards).sum(), self.steps)

	def discount_reward(self):
		self.cumulative_reward = np.array(self.episode_rewards)
		if self.premature_ending:
			print(bcolors.WARNING + "The episode ended prematurely. Using value estimate for the rest. Consider increaseing buffer size." + bcolors.ENDC)
			self.cumulative_reward[-1] = self.CRTModel.model.predict(np.expand_dims(self.observation, axis=0)).squeeze()

		for i in reversed(range(len(self.cumulative_reward) - 1)):
			self.cumulative_reward[i] += self.cumulative_reward[i + 1] * self.GAMMA

	def get_batch(self):
		tmp_observations, tmp_action_matrices, tmp_predicted_actions = [], [], []
		self.episode_rewards = []

		for step in range(self.BUFFER_SIZE):
			action, action_matrix, predicted_action = self.get_action()
			# * This is where the stepping is done!
			observation, reward, done, info = self.env.step(action)
			self.steps += 1
			self.episode_rewards.append(reward)

			tmp_observations.append(self.observation)
			tmp_action_matrices.append(action_matrix)
			tmp_predicted_actions.append(predicted_action)

			self.observation = observation
		
			if self.steps % self.log_frequency == 0:
				self.log_file.write(f"{self.last_fifty_epsiodes[0]},{self.steps}\n")

			if done:
				self.premature_ending = False
				break
		else:
			self.premature_ending = True

		self.discount_reward()
		self.reset_env()

		obs = np.array(tmp_observations)
		action = np.array(tmp_action_matrices)
		pred = np.array(tmp_predicted_actions) 
		reward = np.expand_dims(self.cumulative_reward, axis=1)

		self.log_information(obs, pred)

		return obs, action, pred, reward
	
	def play(self, MAX_EPISODES, pick_best=True):
		self.play_episodes = 0
		max_values = True
		observation = self.env.reset()
		cum_reward = []
		while self.play_episodes < MAX_EPISODES:
			probability_distribution = self.ACTModel.model.predict([np.expand_dims(observation, axis=0), self.DUMMY_VALUE, self.DUMMY_ACTION]).squeeze()
			if not pick_best:
				action = np.random.choice(self.action_shape, p=np.nan_to_num(probability_distribution))
			else:
				action = np.argmax(probability_distribution)
			
			observation, reward, done, info = self.env.step(action)
			#time.sleep(1/60.0)
			self.env.render()
			cum_reward.append(reward)
			if done:
				self.reset_env()
				self.play_episodes += 1
				print(f"The sum of the rewards is {sum(cum_reward)}.")
				cum_reward = []
				max_values = not max_values

