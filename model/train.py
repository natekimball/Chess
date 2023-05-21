import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Input
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from keras.regularizers import l2
import sys
import util

skiprows = util.get_arg('--skip',0)
nrows = util.get_arg('--nrows',12958036//2)

data = pd.read_csv('data/chessData.csv') #,skiprows=skiprows, nrows=nrows) #12958036 total lines
print(data.head())

X = np.array(data['FEN'].apply(util.fen_to_mat).tolist())
y = data['Evaluation'].apply(util.evaluation_to_int).values
print(X.shape)
# print(f"rows {skiprows}-{skiprows+nrows}")

epochs = int(util.get_arg('--epochs',60))
batch_size = int(util.get_arg('--batch-size',256))

input_shape = (13, 8, 8)
num_filters = 128
num_residual_blocks = 12
dropout_rate = .1
regularization_constant = .01

def build_residual_block(inputs, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(regularization_constant))(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(regularization_constant))(x)
    x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
    return x

def build_value_head(inputs):
    x = tf.keras.layers.Conv2D(1, 1, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(regularization_constant))(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(regularization_constant))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(1, name='value', kernel_initializer='he_normal', kernel_regularizer=l2(regularization_constant))(x)
    return x

def build_model(input_shape, num_filters, num_residual_blocks):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(regularization_constant))(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)
    for _ in range(num_residual_blocks):
        x = build_residual_block(x, num_filters)
    value_head = build_value_head(x)
    model = tf.keras.Model(inputs=inputs, outputs=value_head)
    return model

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-4,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model_checkpoint = ModelCheckpoint('best-model.{epoch:02d}-{val_loss:.2f}', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)

load_dir = util.get_arg('--load-dir',None)
if load_dir:
    model = load_model(load_dir)
else:
    model = build_model(input_shape, num_filters, num_residual_blocks)

model.compile(optimizer=optimizer, loss='mse', metrics='mae')

model.summary()

print(model.predict(X[:10]))

history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[reduce_lr, early_stopping, model_checkpoint])

save_dir = util.get_arg('--save-dir','saved_model')
model.save('keras.'+save_dir)
util.save_signatures(model, save_dir)

print(model.predict(X[:10]))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.png')
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()