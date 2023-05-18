
import tensorflow as tf
from keras.models import load_model
import util

model_path = util.get_arg('--model-path', 'model_v4_w_sigs')
out_dir = util.get_arg('--out-dir', 'model_v5_w_sigs')
model = tf.saved_model.load(model_path)
# model = load_model(model_path)
model = util.read_checkpoint(model)
util.save_signatures(model, out_dir)