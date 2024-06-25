# ReinforceLearning

Car_Racing-v2

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
