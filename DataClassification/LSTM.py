import os
import time
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

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

    trainX = scaler.fit_transform(trainX.reshape(num_train_episodes * window_size, -1)).reshape(num_train_episodes,
                                                                                                window_size, -1)
    testX = scaler.transform(testX.reshape(num_test_episodes * window_size, -1)).reshape(num_test_episodes, window_size,
                                                                                         -1)
    classificationX = scaler.transform(classificationX.reshape(num_classification_episodes * window_size, -1)).reshape(
        num_classification_episodes, window_size, -1)

    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    classificationy = to_categorical(classificationy)

    return np.array(trainX), trainy, np.array(testX), testy, np.array(classificationX), classificationy


# 모델 평가 및 예측 함수
def evaluate_and_predict_model(trainX, trainy, testX, testy, classificationX, classificationy, window_size=8, experiment_name='test_LSTM_1'):
    verbose, epochs, batch_size = 1, 300, 32
    n_timesteps, n_features = window_size, trainX.shape[2]
    n_outputs = trainy.shape[1]
    model = Sequential()
    model.add(LSTM(64, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')

    log_dir = f'logs/{experiment_name}'

    tensorboard_callback = TensorBoard(log_dir=log_dir, profile_batch=0)

    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose,
              validation_data=(testX, testy), callbacks=[early_stop, tensorboard_callback])

    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    predictions = model.predict(classificationX)

    return accuracy, predictions


def count_class_frequencies(predictions):
    class_counts = Counter()

    for prediction in predictions:
        predicted_class = np.argmax(prediction)
        class_counts[predicted_class] += 1

    return class_counts


# run experiment
def run_experiment(window_size=8, experiment_name='test_LSTM_1'):
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
        print(f'Class {class_name}: Predicted {count} times')

    print(f"Total time elapsed: {total_time:.2f} seconds")


# 실험 실행
run_experiment(window_size=8)
