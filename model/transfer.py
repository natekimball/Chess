import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# # Load your model (replace this with the code for your specific model)
# model = ...

# # Save the model as a SavedModel
# tf.saved_model.save(model, 'path/to/saved_model')

# Load the SavedModel
loaded = tf.saved_model.load('saved_model')
print(loaded.signatures)
# _SignatureMap({'serving_default': <ConcreteFunction signature_wrapper(*, conv2d_input) at 0x2B2396307340>})
infer = loaded.signatures['serving_default']

# Convert the SavedModel to a frozen graph
frozen_func = convert_variables_to_constants_v2(infer)
frozen_func.graph.as_graph_def()

# Save the frozen graph to a .pb file
with tf.io.gfile.GFile('saved_model.pb', 'wb') as f:
    f.write(frozen_func.graph.as_graph_def().SerializeToString())
