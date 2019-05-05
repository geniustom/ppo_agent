# -*- coding: utf-8 -*-
import h5py,os,yaml
import numpy as np
import matplotlib.pyplot as plt  
from tensorflow.keras.callbacks import ModelCheckpoint


class GModel():
	def __init__(self,model=None,path='model.h5'):
		self.path=path
		self.model=model
		self.checkpoint=None
		self.last_epoch = -1
		self.last_meta = {}
		if model!=None:
			try:
				self.last_epoch, self.last_meta = self.get_last_status()
			except Exception as e: 
				print(str(e))
				self.model.save(self.path)


	def load_meta(self,model_fname):
		"""
		Load meta configuration :return: meta info
		"""
		meta = {}
		with h5py.File(self.path, 'r') as f:
			meta_group = f['meta']
			meta['training_args'] = yaml.load(meta_group.attrs['training_args'])
			for k in meta_group.keys():
				meta[k] = list(meta_group[k])
				
		return meta
	
	def get_last_status(self):
		last_epoch = -1
		last_meta = {}
		if os.path.exists(self.path):
			self.model.load_weights(self.path)
			last_meta = self.load_meta(self.path)
			last_epoch = last_meta.get('epochs')[-1]
			print(last_meta,last_epoch)
		return last_epoch, last_meta

	def show_train_history(train_history, train, validation):  
		plt.plot(train_history.history[train])  
		plt.plot(train_history.history[validation])  
		plt.title('Train History')  
		plt.ylabel(train)  
		plt.xlabel('Epoch')  
		plt.legend(['train', 'validation'], loc='upper left')  
		plt.show()  
		
		
		

class MetaCheckpoint(ModelCheckpoint):
	def __init__(self, filepath, monitor='val_loss', verbose=0,
				 save_best_only=False, save_weights_only=False,
				 mode='auto', period=1, training_args=None, meta=None):

		super(MetaCheckpoint, self).__init__(filepath,
											 monitor=monitor,
											 verbose=verbose,
											 save_best_only=save_best_only,
											 save_weights_only=save_weights_only,
											 mode=mode,
											 period=period)

		self.filepath = filepath
		self.new_file_override = True
		self.meta = meta or {'epochs': [], self.monitor: []}

		if training_args:
			self.meta['training_args'] = training_args

	def on_train_begin(self, logs={}):
		if self.save_best_only:
			if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
				self.best = max(self.meta[self.monitor], default=-np.Inf)
			else:
				self.best = min(self.meta[self.monitor], default=np.Inf)

		super(MetaCheckpoint, self).on_train_begin(logs)

	def on_epoch_end(self, epoch, logs={}):
		# 只有在‘只保存’最优版本且生成新的.h5文件的情况下
		if self.save_best_only:
			current = logs.get(self.monitor)
			#print (current)
			if self.monitor_op(current, self.best):
				self.new_file_override = True
			else:
				self.new_file_override = False

		super(MetaCheckpoint, self).on_epoch_end(epoch, logs)

		# Get statistics
		self.meta['epochs'].append(epoch)
		for k, v in logs.items():
			# Get default gets the value or sets (and gets) the default value
			self.meta.setdefault(k, []).append(v)

		# Save to file
		filepath = self.filepath.format(epoch=epoch, **logs)

		if self.new_file_override and self.epochs_since_last_save == 0:
			# 只有在‘只保存’最优版本且生成新的.h5文件的情况下 才会继续添加meta
			with h5py.File(filepath, 'r+') as f:
				meta_group = f.create_group('meta')
				meta_group.attrs['training_args'] = yaml.dump(
					self.meta.get('training_args', '{}'))
				meta_group.create_dataset('epochs', data=np.array(self.meta['epochs']))
				for k in logs:
					meta_group.create_dataset(k, data=np.array(self.meta[k]))


