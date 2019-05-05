import tensorflow as tf
import os
#import model as mm

# python – Keras：如何保存模型并继续培训？ https://codeday.me/bug/20180921/257413.html
# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
import numpy as np


from statistics import mean
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,Flatten,Lambda,concatenate,add
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from ppo.ppo_base import PPOAgentBase
from ppo.losses import proximal_policy_optimization_loss, EarlyStopByKL

from scipy.stats import entropy
import ppo.ai_lib.model as am

def null_option(t1,t2):
	l1=Dense(1)(t1)
	l2=Dense(1)(t2)
	l3=add([l1,l2])
	return l3


class PPO(PPOAgentBase):        
	def __init__(self, env, experiment_name,
						actor_model=None, 
						critic_model=None,
						LEARNING_RATE=5e-4, 
						LEARNING_RATE_TARGET=5e-5, 
						LOSS_CLIPPING=0.2,
						EARLY_STOPPING_KL_AMOUNT=0.01, 
						ENTROPY_LOSS=0.01, 
						GAMMA=0.99, 
						BATCH_SIZE=32, 
						BUFFER_SIZE=1024, 
						EPOCHS=5, 
						TRAINING_STEP_LENGTH=1e4):
		super().__init__(env=env, experiment_name=experiment_name,
								  actor_model=actor_model, 
								  critic_model=critic_model,
								  LEARNING_RATE=LEARNING_RATE, 
								  LEARNING_RATE_TARGET=LEARNING_RATE_TARGET,
								  LOSS_CLIPPING=LOSS_CLIPPING, 
								  EARLY_STOPPING_KL_AMOUNT=EARLY_STOPPING_KL_AMOUNT, 
								  ENTROPY_LOSS=ENTROPY_LOSS, 
								  GAMMA=GAMMA, BATCH_SIZE=BATCH_SIZE, 
								  BUFFER_SIZE=BUFFER_SIZE, 
								  EPOCHS=EPOCHS, 
								  TRAINING_STEP_LENGTH=TRAINING_STEP_LENGTH,
								  network_style=type(self))

		pass

	

	
	def build_actor_critic(self):
		#这一句很关键，在model实例构造前进行设置，0表示test模式，1表示训练模式
		K.set_learning_phase(1)
		######################## Actor ########################
		old_prediction = Input(shape=(self.action_shape,), name="old_prediction")
		advantage = Input(shape=(1,), name="advantage")
		ob_input_actor = Input(shape=self.observation_shape,  name="observation_actor")
		actor_model=self.actor_model(ob_input_actor)
		out_actions = Dense(self.action_shape, activation='softmax', name='output')(actor_model)
		actor_loss=proximal_policy_optimization_loss(
									advantage=advantage,
									old_prediction=old_prediction, 
									LOSS_CLIPPING=self.LOSS_CLIPPING, 
									ENTROPY_LOSS=self.ENTROPY_LOSS)
		actor = Model(inputs=[ob_input_actor, advantage, old_prediction],  outputs=[out_actions])
		actor.compile(optimizer=Adam(lr=self.LEARNING_RATE), loss=actor_loss)
		######################## Critic ########################
		ob_input_critic = Input(shape=self.observation_shape,  name="observation_critic")
		critic_model=self.critic_model(ob_input_critic)
		out_value = Dense(1, name='critic')(critic_model)
		critic = Model(inputs=[ob_input_critic], outputs=[out_value])
		critic.compile(optimizer=Adam(lr=self.LEARNING_RATE),loss="mse")
		#######################################################
	
		with open(os.path.join(self.result_dir_path, 'summary.txt'),'a+') as fh:
			# Pass the file handle in as a lambda function to make it callable
			actor.summary(print_fn=lambda x: fh.write(x + '\n'))
			critic.summary(print_fn=lambda x: fh.write(x + '\n'))


		plot_model(actor, to_file=os.path.join(self.result_dir_path, 'actor.png'),show_shapes=True)
		plot_model(critic, to_file=os.path.join(self.result_dir_path, 'critic.png'),show_shapes=True)

		return actor, critic


	def learn(self):
		while self.steps < self.TRAINING_STEP_LENGTH:
			obs, action, pred, reward = self.get_batch()
			old_prediction = pred
			pred_values = self.critic.predict(obs)
			advantage = reward - pred_values
			
			# Normalize the advantage
			advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
			actor_callbacks = [EarlyStopByKL(self.EARLY_STOPPING_KL_AMOUNT, obs, advantage, old_prediction, action,self.last_kl_divergence)]
			self.actor_loss =self.ACTModel.model.fit([obs, advantage, old_prediction], [action], batch_size=self.BATCH_SIZE, 
											shuffle=True, epochs=self.EPOCHS, callbacks=actor_callbacks, verbose=False,
											initial_epoch=self.ACTModel.last_epoch + 1)		
			self.critic_loss= self.CRTModel.model.fit([obs], [reward], batch_size=self.BATCH_SIZE, 
											shuffle=True, epochs=self.EPOCHS, verbose=False, initial_epoch=self.CRTModel.last_epoch + 1)		
			new_lr = self.get_new_learning_rate()
			K.set_value(self.actor.optimizer.lr, new_lr) 
			K.set_value(self.critic.optimizer.lr, new_lr) 
			
			self.gradient_steps += 1




'''
	def learn(self):
		while self.steps < self.TRAINING_STEP_LENGTH:
			obs, action, pred, reward = self.get_batch()
			old_prediction = pred
			pred_values = self.critic.predict(obs)
			advantage = reward - pred_values
			
			# Normalize the advantage
			advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
			# Early Stopping
			#actor_ckpt = ModelCheckpoint("./best_actor.h5", monitor='loss', verbose=0, save_best_only=True, mode='min')
			#critic_ckpt = ModelCheckpoint("./best_critic.h5", monitor='loss', verbose=0, save_best_only=True, mode='min')

			actor_ckpt = am.MetaCheckpoint("./best_actor.h5", monitor='loss',save_weights_only=False, save_best_only=False,verbose=0, meta=self.ACTModel.last_meta)
			actor_callbacks = [EarlyStopByKL(self.EARLY_STOPPING_KL_AMOUNT, obs, advantage, old_prediction, action,self.last_kl_divergence)]
			critic_ckpt = am.MetaCheckpoint("./best_critic.h5", monitor='loss',save_weights_only=False, save_best_only=False,verbose=0, meta=self.CRTModel.last_meta)
			#critic_callbacks = [critic_ckpt]
			#print(obs.shape,advantage.shape,old_prediction.shape)
			#self.actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=self.BATCH_SIZE, shuffle=True, epochs=self.EPOCHS, callbacks=actor_callbacks, verbose=False)
			#self.critic_loss = self.critic.fit([obs], [reward], batch_size=self.BATCH_SIZE, shuffle=True, epochs=self.EPOCHS,  verbose=False) #, callbacks=critic_callbacks
			self.actor_loss =self.ACTModel.model.fit([obs, advantage, old_prediction], [action], batch_size=self.BATCH_SIZE, 
											shuffle=True, epochs=self.EPOCHS, callbacks=actor_callbacks, verbose=False,
											initial_epoch=self.ACTModel.last_epoch + 1)		
			self.critic_loss= self.CRTModel.model.fit([obs], [reward], batch_size=self.BATCH_SIZE, 
											shuffle=True, epochs=self.EPOCHS, verbose=False, #callbacks=critic_callbacks,
											initial_epoch=self.CRTModel.last_epoch + 1)		
			new_lr = self.get_new_learning_rate()
			K.set_value(self.actor.optimizer.lr, new_lr) 
			K.set_value(self.critic.optimizer.lr, new_lr) 
			
			self.gradient_steps += 1
'''