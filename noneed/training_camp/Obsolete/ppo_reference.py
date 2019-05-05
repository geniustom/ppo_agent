from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
import gym
import time
import datetime
import os, os.path

# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
import numpy as np

import gym
import collections
from statistics import mean

from keras.utils import plot_model

import numba as nb
from tensorboardX import SummaryWriter

class PPOAgentBase(ABC):
    def __init__(self, env, experiment_name, 
                            LEARNING_RATE=2.5e-4, 
                            LEARNING_RATE_TARGET=0, 
                            LOSS_CLIPPING=0.1, 
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
        self.ENTROPY_LOSS = ENTROPY_LOSS
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.EPOCHS = EPOCHS
        self.TRAINING_STEP_LENGTH = TRAINING_STEP_LENGTH

        # How often to run a validation episode, which is an episode where the agent
        # only takes the best action instead of sampling
        self.VALIDATION_FREQUENCY = 100

        if self.LEARNING_RATE_TARGET:
            self.LEARNING_RATE_DECAY = self.TRAINING_STEP_LENGTH **(1/float(self.LEARNING_RATE/self.LEARNING_RATE_TARGET))
        else:
            self.LEARNING_RATE_DECAY = 1          

        self.last_fifty_epsiodes = collections.deque(50*[0], 50)

        self.observation = self.env.reset()
        
        self.validation_episode = False
        self.episode_rewards = []

        self.result_dir_path = self.get_summary_path(experiment_name)
        self.writer = SummaryWriter(self.result_dir_path)
        
        self.actor, self.critic = self.build_actor_critic()

        self.gradient_steps = 0
        self.episode = 0
        self.steps = 0

        # These are used when you are prediction the action.
        self.DUMMY_ACTION = np.zeros((1, self.action_shape))
        self.DUMMY_VALUE = np.zeros((1, 1))


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
        
    @abstractmethod
    def build_actor_critic(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    def reset_env(self):
        self.episode += 1
        if self.episode % self.VALIDATION_FREQUENCY == 0:
            self.validation_episode = True
        else:
            self.validation_episode = False
        self.observation = self.env.reset()
        self.episode_rewards = []

    def get_action(self):
        probability_distribution = self.actor.predict([np.expand_dims(self.observation, axis=0), self.DUMMY_VALUE, self.DUMMY_ACTION]).squeeze()
        if not self.validation_episode:
            action = np.random.choice(self.action_shape, p=np.nan_to_num(probability_distribution))
        else:
            action = np.argmax(probability_distribution)
        
        action_matrix = np.zeros(self.action_shape)
        action_matrix[action] = 1
        return action, action_matrix, probability_distribution

    def discount_reward(self):
        if not self.validation_episode:
            self.writer.add_scalar('Episode Reward', np.array(self.episode_rewards).sum(), self.steps)
            self.last_fifty_epsiodes.appendleft(sum(self.episode_rewards))
            print(f"Episode {self.episode} reward {np.array(self.episode_rewards).sum():.2f} with step {self.steps} and average: {mean(self.last_fifty_epsiodes):.2f}")
        else:
            self.writer.add_scalar('Validation episode reward', np.array(self.episode_rewards).sum(), self.steps)
            print(f'Validation Episode reward: {np.array(self.episode_rewards).sum():.2f} ', )

        # Todo: Check this math. 
        for i in reversed(range(len(self.episode_rewards) - 1)):
            self.episode_rewards[i] += self.episode_rewards[i + 1] * self.GAMMA

    def get_batch(self):
        batch = [[], [], [], []]

        tmp_observations, tmp_action_matrices, tmp_predicted_actions = [], [], []
        while len(batch[0]) < self.BUFFER_SIZE:
            action, action_matrix, predicted_action = self.get_action()

            # * This is where the stepping is done!
            observation, reward, done, info = self.env.step(action)
            self.steps += 1

            # if self.validation_episode:
            #     self.env.render()
            self.episode_rewards.append(reward)

            tmp_observations.append(self.observation)
            tmp_action_matrices.append(action_matrix)
            tmp_predicted_actions.append(predicted_action)

            self.observation = observation

            if done:
                self.discount_reward()
                if self.validation_episode is False:
                    for i in range(len(tmp_observations)):
                        obs, action, pred = tmp_observations[i], tmp_action_matrices[i], tmp_predicted_actions[i]
                        r = self.episode_rewards[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_observations, tmp_action_matrices, tmp_predicted_actions = [], [], []
                self.reset_env()

        obs = np.array(batch[0])
        action = np.array(batch[1])
        pred = np.array(batch[2]) 
        reward = np.reshape(np.array(batch[3]), (len(batch[3]), 1))
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