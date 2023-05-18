import tensorflow as tf
from keras.models import load_model
import util

output_dir = 'saved_model_t'
# model = load_model('model_v4')

# util.save_signatures(model, output_dir)
model = tf.saved_model.load(output_dir)
print(model.signatures)

checkpoint = tf.train.Checkpoint(model)
checkpoint.read("training_checkpoints/ckpt").assert_consumed()

# now what