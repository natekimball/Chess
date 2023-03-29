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

skiprows = 0
nrows = 100000
data = pd.read_csv('data/chessData.csv', nrows) #12958036 total lines
# data.columns = ['FEN', 'Evaluation']
print(f"rows {skiprows}-{nrows+skiprows}")

print(data.head())

data['FEN'] = data['FEN'].apply(util.fen_to_mat)
data['Evaluation'] = data['Evaluation'].apply(util.evaluation_to_int)
print(data.head())
X = data['FEN'].values
y = data['Evaluation'].values

print(np.array(X).shape)

# Parameters
input_shape = (13, 8, 8)
num_filters = 256
num_residual_blocks = 10

# Input layer
input_layer = Input(shape=input_shape)

# Initial convolutional layer
x = Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')(input_layer)

# Residual blocks
for _ in range(num_residual_blocks):
    y = Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')(x)
    y = Conv2D(filters=num_filters, kernel_size=3, padding='same')(y)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

## Policy head
# policy_head = Conv2D(filters=73, kernel_size=1, padding='same', activation='relu')(x)
# policy_head = Flatten()(policy_head)
# policy_head = Dense(8 * 8 * 73, activation='softmax', name='policy_output')(policy_head)

# Value head
value_head = Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')(x)
value_head = Flatten()(value_head)
value_head = Dense(256, activation='relu')(value_head)
value_head = Dense(1, activation='tanh', name='value_output')(value_head)

# Create the model
model = Model(inputs=input_layer, outputs=value_head)

learning_rate = 2e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='mse', metrics='mae')

model.summary()

# model = load_model('saved_model')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

history = model.fit(X, y, epochs=20, batch_size=64, validation_split=.1, callbacks=[reduce_lr, early_stopping])

print(history)

model.save('saved_model')
util.save_frozen(model)