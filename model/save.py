import tensorflow as tf
from keras.models import load_model
import util

output_dir = util.get_arg('--output-dir','model_v4_w_sigs')
load_dir = util.get_arg('--load_dir', 'model_v4')
model = load_model(load_dir)

util.save_signatures(model, output_dir)
model = tf.saved_model.load(output_dir)
print(model.signatures)