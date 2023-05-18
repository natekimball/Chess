import tensorflow as tf
from keras.models import load_model
import util

output_dir = 'model_v4_w_sigs'
model = load_model('model_v4')

util.save_signatures(model, output_dir)
model = tf.saved_model.load(output_dir)
print(model.signatures)