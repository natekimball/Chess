import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Input
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
import util

# skiprows = 297000#27000
skiprows=0
nrows = 300000
# nrows=100
data = pd.read_csv('data/chessData.csv', nrows=nrows) #12958036 total lines
data.columns = ['FEN', 'Evaluation']
print(f"rows {skiprows}-{nrows+skiprows}")

print(data.head())

data['FEN'] = data['FEN'].apply(util.fen_to_mat)
data['Evaluation'] = data['Evaluation'].apply(util.evaluation_to_int)
print(data.head())
X = np.array(data['FEN'].values.tolist())
y = np.array(data['Evaluation'].values.tolist())
print(X.shape)

epochs = 30
batch_size = 128

# model params
input_shape = (13, 8, 8)
num_filters = 256
num_residual_blocks = 19

def build_residual_block(inputs, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(.25)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(.25)(x)
    x = tf.keras.layers.Add()([x, inputs])
    x = tf.keras.layers.ReLU()(x)
    return x

def build_value_head(inputs):
    x = tf.keras.layers.Conv2D(1, 1, use_bias=False, kernel_initializer='he_normal')(inputs)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(.25)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, use_bias=False, kernel_initializer='he_normal')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(.25)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(1, name='value', kernel_initializer='he_normal')(x)
    return x

def build_model(input_shape, num_filters, num_residual_blocks):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(.25)(x)
    # x = MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.ReLU()(x)
    for _ in range(num_residual_blocks):
        x = build_residual_block(x, 256)
    value_head = build_value_head(x)
    model = tf.keras.Model(inputs=inputs, outputs=value_head)
    return model

initial_learning_rate = 1e-4
decay_rate = 0.96
decay_steps = len(X)*.9 // batch_size

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps,
    decay_rate,
    staircase=True,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model_checkpoint = ModelCheckpoint('best-model.{epoch:02d}-{val_loss:.2f}', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

model = build_model(input_shape, num_filters, num_residual_blocks)
# model = load_model('model_v4')

model.compile(optimizer=optimizer, loss='mse', metrics='mae')

# model.summary()

print(model.predict(X[:5]))

history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[reduce_lr, early_stopping, model_checkpoint])

model.save('model_v5')
util.save_signatures(model, 'saved_model')

print(model.predict(X[:5]))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# util.save_frozen(model)
