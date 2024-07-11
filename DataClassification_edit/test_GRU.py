import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 데이터 로딩 함수
def load_file(filepath, window_size=8):
    with open(filepath, 'r') as file:
        data = [list(map(float, line.strip().split(','))) for line in file]
    episodes = []
    for i in range(0, len(data) - window_size + 1, window_size):
        episodes.append(data[i:i + window_size])
    return np.array(episodes)

# GRU 모델 생성 함수
def build_gru_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.GRU(64, return_sequences=True)(inputs)
    x = layers.GRU(32)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# 메인 코드
if __name__ == "__main__":
    # 데이터 로딩
    dqn_data = load_file('C:/Users/Dohc/PycharmProjects/test/Real/data/DQN.txt')
    ppo_data = load_file('C:/Users/Dohc/PycharmProjects/test/Real/data/PPO.txt')
    keyboard_data = load_file('C:/Users/Dohc/PycharmProjects/test/Real/data/keyboard.txt')

    # 레이블 생성
    dqn_labels = np.zeros(len(dqn_data))
    ppo_labels = np.ones(len(ppo_data))
    keyboard_labels = np.full(len(keyboard_data), 2)

    # 데이터 결합
    X = np.concatenate([dqn_data, ppo_data, keyboard_data], axis=0)
    y = np.concatenate([dqn_labels, ppo_labels, keyboard_labels])

    # 데이터 셔플 및 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 데이터 정규화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # 모델 생성
    input_shape = X_train.shape[1:]
    model = build_gru_model(input_shape)

    # 모델 컴파일
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["sparse_categorical_accuracy"],
    )

    # 모델 요약
    model.summary()

    # 콜백 설정
    callbacks = [
        keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
    ]

    # 모델 훈련
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
    )

    # 모델 평가
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")

    # 학습 곡선 그리기
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 예측 수행
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # 클래스 이름 정의
    class_names = ['DQN', 'PPO', 'Keyboard']

    # 혼동 행렬 계산 및 시각화
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 실제 값과 예측 값 비교 (처음 50개 샘플)
    plt.figure(figsize=(15, 5))
    plt.plot(y_test[:50], 'bo-', label='Actual')
    plt.plot(y_pred_classes[:50], 'ro-', label='Predicted')
    plt.title('Actual vs Predicted Values (First 50 samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.yticks([0, 1, 2], class_names)
    plt.legend()
    plt.grid(True)
    plt.show()

    # 정확도, 오차 샘플 출력
    print("\nSample comparisons (Actual vs Predicted):")
    for i in range(20):  # 처음 20개 샘플에 대해 출력
        actual = int(y_test[i])
        predicted = y_pred_classes[i]
        print(f"Sample {i+1}: Actual = {class_names[actual]}, Predicted = {class_names[predicted]}",
              "✓" if actual == predicted else "✗")

    # 클래스별 정확도 계산
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"Accuracy for {class_names[i]}: {acc:.4f}")