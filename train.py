import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split


data = pd.read_csv('chessData.csv', nrows=99999) #skiprows=999999
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
            match l.lower():
                case 'q':
                    if l.isupper():
                        mat[0][i][j] = 1
                    else:
                        mat[1][i][j] = 1
                case 'k':
                    if l.isupper():
                        mat[2][i][j] = 1
                    else:
                        mat[3][i][j] = 1
                case 'r':
                    if l.isupper():
                        mat[4][i][j] = 1
                    else:
                        mat[5][i][j] = 1
                case 'b':
                    if l.isupper():
                        mat[6][i][j] = 1
                    else:
                        mat[7][i][j] = 1
                case 'n':
                    if l.isupper():
                        mat[8][i][j] = 1
                    else:
                        mat[9][i][j] = 1
                case 'p':
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
            match char:
                case 'K':
                    mat[12, 7, 7] = 1
                case 'k':
                    mat[12, 0, 7] = 1
                case 'Q':
                    mat[12, 7, 0] = 1
                case 'q':
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
        return int(evaluation[1:])
    return int(evaluation)
    

data['FEN'] = data['FEN'].apply(fen_to_mat)
data['Evaluation'] = data['Evaluation'].apply(evaluation_to_int)
print(data.head())
X = data['FEN'].values
y = data['Evaluation'].values

print(data['FEN'][0])
X_train, X_test, y_train, y_test = [i.tolist() for i in train_test_split(X, y, test_size=0.2, random_state=42)]
print(np.array(X_train).shape)

model = Sequential([
    Conv2D(
        filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(13, 8, 8)),
    BatchNormalization(),
    Conv2D(
        filters=32, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=2),
    Conv2D(
        filters=64, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(
        filters=64, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    # MaxPooling2D(pool_size=2),
    Conv2D(
        filters=128, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(
        filters=128, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(
        filters=256, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(
        filters=256, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    # MaxPooling2D(pool_size=2),
    Conv2D(
        filters=512, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(
        filters=512, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(
        filters=1024, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(
        filters=1024, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    # MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(1024, activation='relu'),
    # Dense(units=4096, activation='relu'),
    Dense(1, activation='tanh')
])

# model = Sequential([
#     Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(13, 8, 8)),
#     BatchNormalization(),
#     Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=2),
#     Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=2),
#     Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=2),
#     Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=2),
#     Flatten(),
#     Dense(units=4096, activation='relu'),
#     Dense(units=4096, activation='relu'),
#     Dense(units=1, activation='tanh')
# ])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32)

z = model.evaluate(X_test, y_test, verbose=2)

model.save('model.h5')




# model = keras.Sequential([
#     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(13,8,8)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')
    
#     # tf.keras.layers.Flatten(input_shape=(12, 8, 8)),
#     # tf.keras.layers.Convolution2D(64, (3,3), activation='relu'),
#     # tf.keras.layers.Dense(256, activation='relu'),
#     # tf.keras.layers.Convolution2D(64, (3,3), activation='relu'),
#     # tf.keras.layers.Dropout(0.2),
#     # tf.keras.layers.Convolution2D(64, (3,3), activation='relu'),
#     # tf.keras.layers.Dense(256, activation='relu'),
#     # tf.keras.layers.Dropout(0.2),
#     # tf.keras.layers.Dense(256, activation='relu'),
#     # tf.keras.layers.Convolution2D(64, (3,3), activation='relu'),
#     # tf.keras.layers.Dense(256, activation='relu'),
#     # tf.keras.layers.Convolution2D(64, (3,3), activation='relu'),
#     # tf.keras.layers.Dense(256, activation='relu'),
#     # tf.keras.layers.Convolution2D(64, (3,3), activation='relu'),
#     # tf.keras.layers.Dense(256, activation='relu'),
#     # tf.keras.layers.Convolution2D(64, (3,3), activation='relu'),
#     # tf.keras.layers.Dense(256, activation='relu'),
#     # tf.keras.layers.Dense(1, activation='softmax')
    
# ])

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'], learning_rate=0.001)

# model.fit(X_train,y_train, epochs=10, validation_data=(X_test, y_test))

# model.evaluate(X_test,y_test, verbose=2)


# pgn = open("lichess_db_standard_rated_2018-01.pgn")
# games = []
# while True:
#     game = chess.pgn.read_game(pgn)
#     if game is None:
#         break
#     games.append(game)

# # Convert the dataset to input tensors and target outputs
# inputs = []
# policies = []
# values = []
# for game in games:
#     board = game.board()
#     for move in game.mainline_moves():
#         encoded_board = encode_board(board)
#         inputs.append(encoded_board)
#         policy = np.zeros(73)
#         policy[chess.Move.from_uci(str(move)).uci()] = 1
#         policies.append(policy)
#         values.append(get_value(board, game.headers["Result"]))
#         board.push(move)
# inputs = np.array(inputs)
# policies = np.array(policies)
# values = np.array(values)

# # Split the dataset into training and validation sets
# num_samples = len(inputs)
# train_indices = np.random.choice(num_samples, int(num_samples * 0.9), replace=False)
# val_indices = np.array([i for i in range(num_samples) if i not in train_indices])
# train_inputs = inputs[train_indices]
# train_policies = policies[train_indices]
# train_values = values[train_indices]
# val_inputs = inputs[val_indices]
# val_policies = policies[val_indices]
# val_values = values[val_indices]


# # def policy_loss(target_policies, predicted_policies):
# #     return tf.keras.losses.categorical_crossentropy(target_policies, predicted_policies)

# # def value_loss(target_values, predicted_values):
# #     return tf.keras.losses.mean_squared_error(target_values, predicted_values)

# # Define the optimizer
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# # Build and compile the model
# model = build_model()
# model.compile(optimizer=optimizer, loss=[policy_loss_fn, value_loss_fn], metrics=['accuracy'])