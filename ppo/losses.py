from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow.keras
import numpy as np

# Todo: Check this math. 
# Loss functions
def proximal_policy_optimization_loss(advantage, old_prediction, LOSS_CLIPPING, ENTROPY_LOSS):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        ratio = prob/(old_prob + 1e-8)
        return -K.mean(K.minimum(ratio * advantage, K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * (prob * K.log(prob + 1e-10)))
    return loss

class EarlyStopByKL(tensorflow.keras.callbacks.Callback):
    def __init__(self, target, obs, advantage, old_prediction, action, last_kl_divergence):
        super(tensorflow.keras.callbacks.Callback, self).__init__()
        self.target = target
        self.obs = obs
        self.advantage = advantage
        self.old_prediction = old_prediction
        self.action = action
        self.last_kl_divergence = last_kl_divergence

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.obs, self.advantage, self.old_prediction])
        prob = self.action * y_pred
        old_prob = self.action * self.old_prediction
        kl_divergence = np.abs(np.mean(old_prob - prob))
        self.last_kl_divergence["value"] = kl_divergence
        if kl_divergence > self.target:
            self.model.stop_training = True
