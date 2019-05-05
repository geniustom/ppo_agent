from ppo.ppo_variants import PPO
from tensorflow.keras.layers import Dense
import gym


def build_model(mi):
	m = Dense(128, activation='relu')(mi)
	m = Dense(128, activation='relu')(m)
	return m


env = gym.make("CartPole-v1")	#CartPole-v0 #Breakout-v0
agent = PPO(env, experiment_name="Gamma999",actor_model=build_model, critic_model=build_model, GAMMA=.999, TRAINING_STEP_LENGTH=2e8)
agent.learn()
#agent.play(MAX_EPISODES=1000)
