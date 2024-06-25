# ReinforceLearning

- 본 연구는 OpenAI Gym의 Car_Racing-v2환경에서 Stable_baselines3의 DQN, PPO 알고리즘으로 학습된 자율주행 agent와 사람이 키보드로 조작한 agent를 구분하는 것을 목표로 함
- 현재 연구에선 PPO, DQN, Keyboard agent를 각각 예측하는 모델의 성능이 76% 정도의 accuracy가 나옴

## CarRacing_Video
<img width="80%" src="https://github.com/DoHyeonCheol/CarRacing-v2/assets/108172386/e1eb47ac-8118-4ec6-bba0-6271d9b4f037.gif"/>

### Required Libraries

- gym
- stable_baselines3
- pygame
- matplotlib
- tensorboard
- swig
- pyglet
- mpi4py

### Version Information

- gym  v0.26.2
- stable_baselines3  v2.2.1
- pygame  v2.1.0
- matplotlib  v3.7.5
- tensorboard  v2.10.0
- swig  v4.2.1
- pyglet  v2.0.14
- mpi4py  v4.0.0


PPO Algorithm
모델 저장 경로 설정 및 평가용 환경 생성
- EvalCallback을 초기화하여 50,000 timestep마다 5개의 에피소드로 모델을 평가하고,
- 최고 성능모델을 저장하도록 설정
- 1,000,000 timestep 동안 PPO 알고리즘으로 학습을 진행 
- (callback=eval_callback을 통해 앞서 설정한 EvalCallback을 사용)

PPO VS DQN

- Action Space
  PPO : Continuous Action Space
  DQN : Discrete Action Space
  CarRacing-v2 환경은 기본적으로 Continuous Action Space를 가지기에 
  DQN 알고리즘을 학습할 때 ‘continuous=False’를 추가 해주는 것이 필수

- Rendering

  PPO : Policy-based 알고리즘으로, 환경의 상태 벡터를 직접 입력으로 받아 Policy를 학습
  DQN : Q-Learning 기반의 Value-based 알고리즘으로, 이미지 관측치를 잘 처리할 수 있음
  DQN알고리즘으로 학습하기 위해서 render_mode=’rgb_array’를 설정하여 
  환경 렌더링 결과를 Numpy 배열로 얻을 수 있음
  반면, PPO 알고리즘으로 학습할 때는 상태 벡터를 직접 입력받기에 별도의 렌더링이 필요 없음
        

- EvalCallback

  PPO : Policy-based 알고리즘으로, 별도의 평가 환경이 필요함
  EvalCallback을 사용해 일정 간격으로 현재의 Policy를 평가 환경에서 평가하고,
  최고 성능 모델을 저장함 → 성능 추이를 모니터링 할 수 있음
   DQN : 경험 재현 버퍼를 사용해, 과거의 경험(상태, 행동, 보상, 다음 상태)을 저장
  학습 시 버퍼에서 무작위로 샘플을 추출하여 Q 함수를 업데이트
  (과거의 다양한 상태를 평가하게 되므로 별도의 평가 환경이 필요X)

- Log파일
  
  로그는 x좌표, y좌표, 속도, 조향각, 가속도 총 5개의 features를 가짐
  좌표들은 환경의 정중앙을 (0,0)으로 설정하며, 우측상단을 (1,1)으로하며, 좌측하단을 (-1,-1)으로 설정
  PPO, DQN, Keyboard 각 에이전트마다 주행 패턴이 다르기에 방향키로 조작하는 것이 다르기에 주행 패턴을 파악하기 위해 5개의 feature을 선택

- DataClassification
  
  시계열 데이터 분류에 좋은 성능을 보이는 딥러닝 모델들 중 CNN, LSTM, ConvLSTM, GRU, Trasformer 모델을 선택하여 분류

  CNN(Convolution Neural Network)은 일반적으로 이미지 처리에 사용되지만, 1D 합성곱 연산을 통해 시계열 데이터에서도 효과적으로 특징을 추출할 수 있음
  인접한 시점 간의 국소적 패턴을 인식하는 데 탁월하며, 시계열 데이터의 변동성에 강한 특징을 추출할 수 있음
  본 연구에서는 CNN모델에 Conv1D 레이어 2개, MaxPooling1D 레이어 1개, Flatten 레이어 1개, Dense 레이어 2개를 사용

  LSTM(Long Short-Term Memory)은 시계열 데이터의 장기 의존성을 포착하는 데 탁월한 성능을 보임
  LSTM의 게이트 구조는 장기 기억을 효과적으로 유지하고 업데이트할 수 있어, 시계열 데이터에서 중요한 패턴을 인식할 수 있음
  본 연구에서는 LSTM 모델에 LSTM 레이어 1개, Dense 레이어 2개를 사용하였음

  ConvLSTM(Convolutional LSTM)은 CNN과 LSTM의 장점을 결합한 모델로, 시계열 데이터에서 시공간적 특징을 동시에 학습할 수 있음 ConvLSTM은 복잡한 시공간 패턴 인식에 효과적
  본 연구에서는 ConvLSTM 모델에 ConvLSTM2D 레이어 1개, Flatten 레이어 1개, Dense 레이어 2개를 사용

  GRU는 LSTM과 유사하게 시계열 데이터의 장기 의존성을 포착할 수 있는 순환 신경망 모델
  하지만, LSTM 모델보다 간소화된 게이트 구조를 가지고 있어 계산 효율성이 높고, 빠른 학습이 가능하다는 장점을 가지고 있음
  본 연구에서는 GRU 모델에 GRU 레이어 1개, Dense 레이어 2개를 사용
  
  Transformer는 어텐션 메커니즘을 기반으로 한 모델로, 시계열 데이터에서 장거리 의존성을 효과적으로 포착할 수 있음
  어텐션 메커니즘을 통해 중요한 시점에 집중하여 정보를 추출할 수 있으며, 병렬처리가 가능하여 빠른 학습속도를 가진다는 장점이 있음
  본 연구에서는 Transformer 모델에 Transformer 인코더 블록 8개, GlobalAveragePooling1D 레이어 1개, Dense 레이어 2개를 사용

  결과는 GRU가 가장 좋은 성능을 가졌으며, 다음으로는 LSTM, CNN, ConvLSTM 순으로 좋은 성능을 보였으며,
  Transformer는 학습이 안되는 문제가 있기에 수정이 필요할 것으로 보임

