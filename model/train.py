import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import util
import model

data = pd.read_csv('data/chessData.csv', skiprows=range(1,199999), nrows=99999) #skiprows=159999
# data.set_index(['FEN', 'Evaluation'])
# data.dropna(inplace=True)

print(data.head())

data['FEN'] = data['FEN'].apply(util.fen_to_mat)
data['Evaluation'] = data['Evaluation'].apply(util.evaluation_to_int)
print(data.head())
X = data['FEN'].values
y = data['Evaluation'].values

# print(data['FEN'][0])
X_train, X_test, y_train, y_test = [i.tolist() for i in train_test_split(X, y, test_size=0.2, random_state=42)]
print(np.array(X_train).shape)

# model = Sequential([
#     Conv2D(
#         filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(13, 8, 8)),
#     BatchNormalization(),
#     Conv2D(
#         filters=32, kernel_size=3, padding='same', activation='relu'),
#     MaxPooling2D(pool_size=1),
#     BatchNormalization(),
#     Conv2D(
#         filters=64, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(
#         filters=64, kernel_size=3, padding='same', activation='relu'),
#     MaxPooling2D(pool_size=1),
#     BatchNormalization(),
#     Conv2D(
#         filters=128, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(
#         filters=128, kernel_size=3, padding='same', activation='relu'),
#     MaxPooling2D(pool_size=1),
#     BatchNormalization(),
#     Conv2D(
#         filters=256, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(
#         filters=256, kernel_size=3, padding='same', activation='relu'),
#     MaxPooling2D(pool_size=1),
#     BatchNormalization(),
#     Conv2D(
#         filters=512, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(
#         filters=512, kernel_size=3, padding='same', activation='relu'),
#     MaxPooling2D(pool_size=2),
#     BatchNormalization(),
#     Conv2D(
#         filters=1024, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(
#         filters=1024, kernel_size=3, padding='same', activation='relu'),
#     MaxPooling2D(pool_size=2),
#     BatchNormalization(),
#     Flatten(),
#     Dense(units=4096, activation='relu'),
#     Dropout(0.25),
#     Dense(units=4096, activation='relu'),
#     Dropout(0.25),
#     Dense(units=2048, activation='relu'),
#     Dropout(0.25),
#     Dense(units=1024, activation='relu'),
#     Dense(1, activation='tanh')
# ])

model = keras.models.load_model('saved_model')

# opt = keras.optimizers.Adam(learning_rate=3e-4)
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.optimizer.lr = 3e-4

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

z = model.evaluate(X_test, y_test, verbose=2)

print(z)

model.save('saved_model')
save_frozen(model)