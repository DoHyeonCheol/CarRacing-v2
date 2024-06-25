from numpy import array
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from collections import Counter
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

scaler = StandardScaler()
log_dir = 'logs'

def load_file(filepath, window_size=8):
    with open(filepath, 'r') as file:
        data_points = [float(x) for line in file for x in line.strip().split(', ')]

    episodes = []
    episode = []
    for data_point in data_points:
        episode.append(data_point)
        if len(episode) == window_size:
            episodes.append(episode)
            episode = []

    if episode:
        episodes.append(episode + [0.0] * (window_size - len(episode)))
    return episodes

# load dataset group
def load_dataset_group(group_files, window_size=8):
    loaded = []
    for file in group_files:
        data = load_file(file, window_size)
        loaded.extend(data)
    return loaded

# split dataset into train, test, and classification sets
def split_dataset(data, train_ratio=0.7, test_ratio=0.2, classification_ratio=0.1):
    num_samples = len(data)
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    classification_data = data[train_size + test_size:]

    return train_data, test_data, classification_data

# load all dataset groups
def load_dataset(prefix='C:/Users/Dohc/PycharmProjects/test/Real/', window_size=8):
    global scaler  # 전역 변수 scaler 사용

    # load sequences
    dqn_data = load_dataset_group([prefix + 'DQN.txt'], window_size)
    ppo_data = load_dataset_group([prefix + 'PPO.txt'], window_size)
    keyboard_data = load_dataset_group([prefix + 'keyboard.txt'], window_size)

    # split datasets into train, test, and classification sets
    train_dqn, test_dqn, classification_dqn = split_dataset(dqn_data)
    train_ppo, test_ppo, classification_ppo = split_dataset(ppo_data)
    train_keyboard, test_keyboard, classification_keyboard = split_dataset(keyboard_data)

    trainX = train_dqn + train_ppo + train_keyboard
    trainy = [0] * len(train_dqn) + [1] * len(train_ppo) + [2] * len(train_keyboard)

    testX = test_dqn + test_ppo + test_keyboard
    testy = [0] * len(test_dqn) + [1] * len(test_ppo) + [2] * len(test_keyboard)

    classificationX = classification_dqn + classification_ppo + classification_keyboard
    classificationy = [0] * len(classification_dqn) + [1] * len(classification_ppo) + [2] * len(classification_keyboard)

    trainX = np.array(trainX)
    testX = np.array(testX)
    classificationX = np.array(classificationX)

    num_train_episodes = trainX.shape[0]
    num_test_episodes = testX.shape[0]
    num_classification_episodes = classificationX.shape[0]

    trainX = scaler.fit_transform(trainX.reshape(num_train_episodes * window_size, -1)).reshape(num_train_episodes, window_size, -1)
    testX = scaler.transform(testX.reshape(num_test_episodes * window_size, -1)).reshape(num_test_episodes, window_size, -1)
    classificationX = scaler.transform(classificationX.reshape(num_classification_episodes * window_size, -1)).reshape(num_classification_episodes, window_size, -1)

    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    classificationy = to_categorical(classificationy)

    return array(trainX), trainy, array(testX), testy, array(classificationX), classificationy

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate=0.1):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-5)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout_rate
    )(x, x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-5)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x + res


def build_transformer_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout_rate=0.1,
    num_classes=3
):
    inputs = keras.Input(shape=input_shape, name='trajectory')
    x = inputs

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def evaluate_and_predict_model(trainX, trainy, testX, testy, classificationX, classificationy, window_size=8, experiment_name='test_transformer_3'):
    verbose, epochs, batch_size = 1, 300, 32
    n_timesteps, n_features = window_size, trainX.shape[2]
    n_outputs = trainy.shape[1]

    model = build_transformer_model(
        input_shape=(n_timesteps, n_features),
        head_size=256,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=8,
        mlp_units=[128, 64],
        dropout_rate=0.1,
        num_classes=n_outputs
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='min')

    log_dir = f'logs/{experiment_name}'

    tensorboard_callback = TensorBoard(log_dir=log_dir, profile_batch=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)

    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose,
              validation_data=(testX, testy), callbacks=[early_stop, tensorboard_callback, reduce_lr])

    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)

    predictions = model.predict(classificationX)

    return accuracy, predictions

def count_class_frequencies(predictions):
    class_counts = Counter()

    for prediction in predictions:
        predicted_class = np.argmax(prediction)
        class_counts[predicted_class] += 1

    return class_counts

# run experiment
def run_experiment(window_size=8, experiment_name='test_transformer_3'):
    trainX, trainy, testX, testy, classificationX, classificationy = load_dataset(window_size=window_size)

    start_time = time.time()

    # evaluate model and predict classification data
    accuracy, predictions = evaluate_and_predict_model(trainX, trainy, testX, testy, classificationX, classificationy, window_size, experiment_name)

    end_time = time.time()
    total_time = end_time - start_time

    print(f'Test accuracy: {accuracy * 100:.2f}%')

    # 예측 결과 출력
    class_counts = count_class_frequencies(predictions)
    for class_label, count in class_counts.items():
        if class_label == 0:
            class_name = "DQN"
        elif class_label == 1:
            class_name = "PPO"
        else:
            class_name = "Keyboard"
        print(f'Class {class_name}: Predicted {count} times')

    print(f"Total time elapsed: {total_time:.2f} seconds")

# 실험 실행
run_experiment(window_size=8)