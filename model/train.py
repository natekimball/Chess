import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow import load_model


data = pd.read_csv('data/chessData.csv', skiprows=79999, nrows=99999) #skiprows=79999
data.dropna(inplace=True)

print(data.head())

def fen_to_mat(fen):
    mat = np.zeros((13, 8, 8), dtype=np.int8)
    fen = fen.split(' ')
    i,j = 0,0
    for l in fen[0]:
        if l == ' ':
            break
        elif l == '/':
            i += 1
            j = 0
        elif l.isdigit():
            j += int(l)
        else:
            r = l.lower()
            if r == 'q':
                if l.isupper():
                    mat[0][i][j] = 1
                else:
                    mat[1][i][j] = 1
            elif r == 'k':
                if l.isupper():
                    mat[2][i][j] = 1
                else:
                    mat[3][i][j] = 1
            elif r == 'r':
                if l.isupper():
                    mat[4][i][j] = 1
                else:
                    mat[5][i][j] = 1
            elif r == 'b':
                if l.isupper():
                    mat[6][i][j] = 1
                else:
                    mat[7][i][j] = 1
            elif r == 'n':
                if l.isupper():
                    mat[8][i][j] = 1
                else:
                    mat[9][i][j] = 1
            elif r == 'p':
                if l.isupper():
                    mat[10][i][j] = 1
                else:
                    mat[11][i][j] = 1
            j += 1
    
    player = fen[1]
    castling_rights = fen[2]
    en_passant = fen[3]
    halfmove_clock = int(fen[4])
    fullmove_clock = int(fen[5])
    
    if en_passant != '-':
        mat[12, ord(en_passant[0]) - ord('a'), int(en_passant[1]) - 1] = 1
    if castling_rights != '-':
        for char in castling_rights:
            if char == 'K':
                mat[12, 7, 7] = 1
            elif char == 'k':
                mat[12, 0, 7] = 1
            elif char == 'Q':
                mat[12, 7, 0] = 1
            elif char == 'q':
                mat[12, 0, 0] = 1
    if player == 'w':
        mat[12, 7, 4] = 1
    else:
        mat[12, 0, 4] = 1
    if halfmove_clock > 0:
        c = 7
        while halfmove_clock > 0:
            mat[12, 3, c] = halfmove_clock%2
            halfmove_clock = halfmove_clock // 2
            c -= 1
            if c < 0:
                break
    if fullmove_clock > 0:
        c = 7
        while fullmove_clock > 0:
            mat[12, 4, c] = fullmove_clock%2
            fullmove_clock = fullmove_clock // 2
            c -= 1
            if c < 0:
                break
    return mat.tolist()

def evaluation_to_int(evaluation):
    if evaluation[0] == '#':
        return int(evaluation[1:])/10
    return int(evaluation)/10
    

data['FEN'] = data['FEN'].apply(fen_to_mat)
data['Evaluation'] = data['Evaluation'].apply(evaluation_to_int)
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

model = load_model('model.h5')

# opt = keras.optimizers.Adam(learning_rate=3e-4)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.optimizer.lr = 3e-4

history = model.fit(X_train, y_train, epochs=20, batch_size=256)

z = model.evaluate(X_test, y_test, verbose=2)

model.save('model.h5')