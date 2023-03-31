import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Input
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import util

# skiprows = 0
# nrows = 50000
skiprows = 50000
nrows = 50000
data = pd.read_csv('data/chessData.csv', skiprows=skiprows, nrows=nrows) #12958036 total lines
# data.columns = ['FEN', 'Evaluation']
print(f"rows {skiprows}-{nrows+skiprows}")

print(data.head())

data['FEN'] = data['FEN'].apply(util.fen_to_mat)
data['Evaluation'] = data['Evaluation'].apply(util.evaluation_to_int)
print(data.head())
X = np.array(data['FEN'].values.tolist())
y = np.array(data['Evaluation'].values.tolist())
print(X.shape)

epochs = 30
batch_size = 64

# model params
input_shape = (13, 8, 8)
num_filters = 256
num_residual_blocks = 12

def build_residual_block(inputs, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.ReLU()(x)
    return x

def build_value_head(inputs):
    x = tf.keras.layers.Conv2D(1, 1, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(1, activation='tanh', name='value')(x)
    return x

def build_model(input_shape, num_filters, num_residual_blocks):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    for _ in range(num_residual_blocks):
        x = build_residual_block(x, 256)
    value_head = build_value_head(x)
    model = tf.keras.Model(inputs=inputs, outputs=value_head)
    return model

# learning_rate = 5e-4
initial_learning_rate = 1e-3
decay_rate = 0.96
decay_steps = len(X)*.9 // batch_size

# Create the ExponentialDecay scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps,
    decay_rate,
    staircase=True,
)

# Create an optimizer using the scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

# model = build_model(input_shape, num_filters, num_residual_blocks)
model = load_model('saved_model')

model.compile(optimizer=optimizer, loss='mse', metrics='mae')

model.summary()

# print(model.predict(X[:10]))

history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[reduce_lr, early_stopping])

model.save('saved_model')

# util.save_frozen(model)
