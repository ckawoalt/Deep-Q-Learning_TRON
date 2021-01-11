import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

GAMMA = 0.9  # 시간할인율
#DDQN
BATCH_SIZE=64


lr = 3e-3
eps = 1e-5
alpha = 0.99

NUM_PROCESSES = 16  # 동시 실행 환경 수
NUM_ADVANCED_STEP = 5 # 총 보상을 계산할 때 Advantage 학습을 할 단계 수

# A2C 손실함수 계산에 사용되는 상수
value_loss_coef = 0.5
entropy_coef = 0.01
policy_loss_coef = 1
max_grad_norm = 0.5

MAP_WIDTH = 10
MAP_HEIGHT = 10

SHOW_ITER = 20

PLAY_WITH_MINIMAX=50

# 리워드 function

slide=0.15

# win, lose, win_separated, win_length_factor

reward_cons1 = [10, -10]

reward_cons2 = [10, -20]

reward_cons3 = [20, -10]
