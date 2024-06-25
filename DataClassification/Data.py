
'''
import numpy as np

def process_data(input_file, output_file):
    # 파일에서 데이터 읽어오기
    with open(input_file, 'r') as f:
        data = np.loadtxt(f, delimiter=',')

    # 첫 번째와 마지막 열을 제외한 데이터 선택
    processed_data = data[:, 1:-1]

    # 결과 파일에 데이터 쓰기 (소수점 이하 5자리 숫자까지만 남기기)
    with open(output_file, 'w') as f:
        np.savetxt(f, processed_data, delimiter=', ', fmt='%.5f')


# 입력 및 출력 파일 경로 지정
input_file = './log/edit/test_PPO.txt'
output_file = './log/test_PPO.txt'

# 데이터 처리 함수 호출
process_data(input_file, output_file)

print("데이터 처리가 완료되었습니다.")
'''

'''
import os
import numpy as np

def process_data(input_file, output_file):
    # 파일에서 데이터 읽어오기
    with open(input_file, 'r') as f:
        data = np.loadtxt(f, delimiter=',')

    # 데이터에 10^5를 곱하여 정수로 변환
    data *= 10**7
    data = data.astype(int)

    # 디렉토리가 없는 경우 생성
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 결과 파일에 데이터 쓰기
    with open(output_file, 'w') as f:
        np.savetxt(f, data, delimiter=', ', fmt='%d')

# 입력 및 출력 파일 경로 지정
input_file = 'C:/Users/Dohc/PycharmProjects/test/Training/Logs/train_PPO.txt'
output_file = 'C:/Users/Dohc/PycharmProjects/test/Training/Logs/edit/train_PPO.txt'

# 데이터 처리 함수 호출
process_data(input_file, output_file)

print("데이터 처리가 완료되었습니다.")
'''


import os

def process_data(input_file, output_file, exclude_keyword):
    # 디렉토리가 없는 경우 생성
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 결과 파일 열기
    with open(output_file, 'w') as output_f:
        # 파일에서 데이터 읽어오기
        with open(input_file, 'r') as input_f:
            for line in input_f:
                # 특정 문구가 포함되지 않은 경우에만 결과 파일에 쓰기
                if exclude_keyword not in line:
                    output_f.write(line)

# 입력 및 출력 파일 경로 지정
input_file = '/Real/train_DQN.txt'
output_file = '/Real/edit/train_DQN.txt'

# 제거할 문구 지정
exclude_keyword = '-0.0, 0.0'

# 데이터 처리 함수 호출
process_data(input_file, output_file, exclude_keyword)

print("데이터 처리가 완료되었습니다.")