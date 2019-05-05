from ppo.ppo_variants import PPO
#from ppo.wrappers  import NormalizeObservations1D

import gym


# Todo: Clip Rate decay.
# Todo: Add CNN.
# Todo: Add the feature extractor.
# Todo: GAE

env = gym.make("CartPole-v1")	#CartPole-v0 #Breakout-v0
#env_normal = NormalizeObservations1D(env)

# save_hyperparameters(["experiment.py"], f"{result_dir_path}/code_snapshot.txt")
#agent = PPO(env_normal, experiment_name="Normalized", TRAINING_STEP_LENGTH=2e5)
#agent.learn()

agent = PPO(env, experiment_name="Gamma999", GAMMA=.999, TRAINING_STEP_LENGTH=2e8)
agent.learn()
agent.play(MAX_EPISODES=1000)
