from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
import gym
import time
import datetime
import os, os.path
import warnings

# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
import numpy as np

import gym
import collections
from statistics import mean

from keras.utils import plot_model
from scipy.stats import entropy

import numba as nb
from tensorboardX import SummaryWriter

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
                            LEARNING_RATE=2.5e-4, 
                            LEARNING_RATE_TARGET=0, 
                            LOSS_CLIPPING=0.2,
                            EARLY_STOPPING_KL_AMOUNT=0.001, 
                            ENTROPY_LOSS=0.01, 
                            GAMMA=0.99, 
                            BATCH_SIZE=32, 
                            BUFFER_SIZE=256, 
                            EPOCHS=4, 
                            TRAINING_STEP_LENGTH=1e4):

        self.ENV_NAME = env.unwrapped.spec.id

        print(f"Got in the {self.ENV_NAME} env.")
        self.env = env
    
        print(f"The action space is: {self.env.unwrapped.action_space}")
        print(f"The observation space is: {self.env.unwrapped.observation_space}")

        # TODO: Add support for continious action space.
        self.action_shape = self.env.unwrapped.action_space.n
        self.observation_shape = self.env.unwrapped.observation_space.shape

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
      

        self.last_fifty_epsiodes = collections.deque(50*[0], 50)

        self.observation = self.env.reset()
        
        self.episode_rewards = []

        self.result_dir_path = self.get_summary_path(experiment_name)
        self.writer = SummaryWriter(self.result_dir_path)
        
        self.actor, self.critic = self.build_actor_critic()

        self.gradient_steps = 0
        self.episode = 0
        self.steps = 0
        self.last_kl_divergence = {"value": 0}

        # These are used when you are prediction the action.
        self.DUMMY_ACTION = np.zeros((1, self.action_shape))
        self.DUMMY_VALUE = np.zeros((1, 1))

        self.max_entropy = entropy([1.0/2.0] * self.action_shape)

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
        probability_distribution = self.actor.predict([np.expand_dims(self.observation, axis=0), self.DUMMY_VALUE, self.DUMMY_ACTION]).squeeze() 
        action = np.random.choice(self.action_shape, p=np.nan_to_num(probability_distribution))

        action_matrix = np.zeros(self.action_shape)
        action_matrix[action] = 1
        return action, action_matrix, probability_distribution

    def log_information(self, observations, predictions, actor_loss, critic_loss):
        self.last_fifty_epsiodes.appendleft(sum(self.episode_rewards))

        if self.premature_ending:
            self.episode_rewards = self.episode_rewards[:-1]
            
        episode_prediction = self.critic.predict(np.expand_dims(observations[0], axis=0)).squeeze()
        episode_reward_sum = np.array(self.episode_rewards).sum()
        value_delta = episode_reward_sum - episode_prediction
        entropy_amount = entropy(np.transpose(predictions)).mean() 

        print(f"* ----------- Episode {self.episode} ----------- ")
        print(f"* Mean 50 Reward : {mean(self.last_fifty_epsiodes):.2f}")
        print(f"* Episode Reward : {episode_reward_sum:.2f}")
        print(f"* Entropy Ratio  : {entropy_amount / self.max_entropy * 100:.2f} %")
        print(f"* Predicted Val  : {episode_prediction:.2f}")
        print(f"* Value Delta    : {value_delta:.2f}")
        print(f"* Steps          : {self.steps}")
        print(f"* Completion     : {self.steps / self.TRAINING_STEP_LENGTH * 100:.2f} %")
        print(f"* KL Divergence  : {self.last_kl_divergence['value']:.4f}")
        print(f"* Current LR     : {self.get_new_learning_rate():.3e}")
        print(f"* ----------------------------------- ") 
        

        self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.steps)
        self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.steps)
        self.writer.add_scalar("Entropy", entropy_amount, self.steps)
        self.writer.add_scalar("Value_delta", value_delta, self.steps)

        self.writer.add_scalar('Episode Reward', np.array(self.episode_rewards).sum(), self.steps)

    def discount_reward(self):
        self.cumulative_reward = np.array(self.episode_rewards)
        if self.premature_ending:
            print(bcolors.WARNING + "The episode ended prematurely. Using value estimate for the rest. Consider increaseing buffer size." + bcolors.ENDC)
            self.cumulative_reward[-1] = self.critic.predict(np.expand_dims(self.observation, axis=0)).squeeze()

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

        return obs, action, pred, reward
    
    def play(self, MAX_EPISODES, pick_best=True):
        self.play_episodes = 0
        max_values = True
        observation = self.env.reset()
        cum_reward = []
        while self.play_episodes < MAX_EPISODES:
            probability_distribution = self.actor.predict([np.expand_dims(observation, axis=0), self.DUMMY_VALUE, self.DUMMY_ACTION]).squeeze()
            if not pick_best:
                action = np.random.choice(self.action_shape, p=np.nan_to_num(probability_distribution))
            else:
                action = np.argmax(probability_distribution)
            
            observation, reward, done, info = self.env.step(action)
            time.sleep(1/60.0)
            self.env.render()
            cum_reward.append(reward)
            if done:
                self.reset_env()
                self.play_episodes += 1
                print(f"The sum of the rewards is {sum(cum_reward)}.")
                cum_reward = []
                max_values = not max_values