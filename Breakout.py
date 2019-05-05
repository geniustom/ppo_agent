from ppo.ppo_variants import PPO
from tensorflow.keras.layers import Dense,Conv2D,Flatten,LeakyReLU
import gym


def build_model(mi):
	m = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(mi)
	m = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(m)
	m = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(m)
	m = Flatten()(m)
	m = Dense(512)(m)
	m = LeakyReLU()(m)
	return m


env = gym.make("MsPacman-v4")	#CartPole-v0 #Breakout-v0
agent = PPO(env, experiment_name="Gamma999",actor_model=build_model, critic_model=build_model, GAMMA=.999, TRAINING_STEP_LENGTH=2e8)
#agent.learn()
agent.play(MAX_EPISODES=1000)
