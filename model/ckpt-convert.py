
import tensorflow as tf
from keras.models import load_model
import util


model_path = 'model_v4'
model = util.read_checkpoint(load_model(model_path), 'training_checkpoints')
util.save_signatures(model, 'model_v5_w_sigs')