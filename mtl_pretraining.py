import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# 환경 설정 및 하이퍼파라미터
DATA_DIR = './data/processed'
MODEL_DIR = './models'
TARGET_CLUSTER = 3 # 사전 학습할 군집 번호

WINDOW_SIZE = 168 # 과거 7일
PREDICT_SIZE = 24 # 미래 1일
BATCH_SIZE = 64
EPOCHS = 300
PATIENCE = 15 # 조기 종료 조건
LEARNING_RATE = 0.001

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")

# 군집 데이터 로드 및 시퀀스 데이터 생성
print(f"\nCluster {TARGET_CLUSTER}의 데이터로 사전 학습 데이터셋 구성 중")

# 데이터 로드
X_tensor = torch.load(f"{DATA_DIR}/multitask_timeseries_top200.pt") # (200, 720, 3)
with open(f"{DATA_DIR}/cluster_assignments.pkl", 'rb') as f:
    cluster_assignments = pickle.load(f)
with open(f"{DATA_DIR}/metadata.pkl", 'rb') as f:
    metadata = pickle.load(f)

cell_ids = metadata['cell_ids']

# TARGET_CLUSTER에 속한 격자들의 인덱스 찾기
cluster_indices = [idx for idx, cell_id in enumerate(cell_ids) if cluster_assignments[cell_id] == TARGET_CLUSTER]
print(f"Cluster {TARGET_CLUSTER}에 속한 격자 수: {len(cluster_indices)}개")

# 슬라이딩 윈도우 생성 함수
def create_multistep_sequences(data, window, predict):
    xs, ys = [], []
    for i in range(len(data) - window - predict + 1):
        xs.append(data[i : (i + window)])
        ys.append(data[(i + window) : (i + window + predict)])
    return xs, ys

all_X, all_y = [], []

# 선택된 격자들의 데이터를 순회하며 시퀀스 추출 및 병합
for idx in cluster_indices:
    cell_data = X_tensor[idx].numpy() # (720, 3)
    xs, ys = create_multistep_sequences(cell_data, WINDOW_SIZE, PREDICT_SIZE)
    all_X.extend(xs)
    all_y.extend(ys)

train_X = torch.FloatTensor(np.array(all_X))
train_y = torch.FloatTensor(np.array(all_y))

print(f"슬라이딩 윈도우 생성 완료")
print(f"학습 데이터(X) 형태: {train_X.shape}") # (Samples, 168, 3)
print(f"타겟 데이터(y) 형태: {train_y.shape}") # (Samples, 24, 3)

train_dataset = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 다중 작업 LSTM 모델 구조 정의 (Multi-Task Learning)
class MultiTaskTrafficLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=3, output_size=24):
        super(MultiTaskTrafficLSTM, self).__init__()

        # Shared Layer (공유 레이어): 3가지 트래픽의 공통된 시공간 특징을 추출
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Specific Layers (작업별 특화 레이어): 각 트래픽별로 24시간 예측값을 따로 출력
        self.fc_sms = nn.Linear(hidden_size, output_size)
        self.fc_call = nn.Linear(hidden_size, output_size)
        self.fc_internet = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (Batch, Window, Features)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :] # 마지막 타임스텝의 은닉 상태 추출

        # 각각의 예측 브랜치로 전달
        out_sms = self.fc_sms(last_hidden)
        out_call = self.fc_call(last_hidden)
        out_internet = self.fc_internet(last_hidden)

        return out_sms, out_call, out_internet

# 이전 연구에서 최적이었던 H64_L3 구조 적용
model = MultiTaskTrafficLSTM(hidden_size=64, num_layers=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 사전 학습 진행
print("\n사전 학습 시작")
best_loss = float('inf')
early_stop_counter = 0
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = []

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward
        pred_sms, pred_call, pred_internet = model(batch_X)

        # 실제값 분리: batch_y shape -> (Batch, 24, 3)
        target_sms = batch_y[:, :, 0]
        target_call = batch_y[:, :, 1]
        target_internet = batch_y[:, :, 2]

        # Multi_Task Loss 계산 (3가지 트래픽의 오차를 모두 더함)
        loss_sms = criterion(pred_sms, target_sms)
        loss_call = criterion(pred_call, target_call)
        loss_internet = criterion(pred_internet, target_internet)

        total_loss = loss_sms + loss_call + loss_internet

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_losses.append(total_loss.item())

    avg_loss = np.mean(epoch_losses)
    loss_history.append(avg_loss)

    # 조기 종료(Early Stopping) 및 모델 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), f"{MODEL_DIR}/pretrained_cluster_{TARGET_CLUSTER}.pth")
    else:
        early_stop_counter += 1

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Total Loss: {avg_loss:.6f} | SMS: {loss_sms.item():.4f}, Call: {loss_call.item():.4f}, Int: {loss_internet.item():.4f}")

    if early_stop_counter >= PATIENCE:
        print(f"\n조기 종료함. (Epoch {epoch+1})")
        break

print(f"\n사전 학습 완료. 최고 성능 가중치 저장됨: {MODEL_DIR}/pretrained_cluster_{TARGET_CLUSTER}.pth")

# 학습 곡선 시각화
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='purple')
plt.title(f'Pre-training Loss Curve (Cluster {TARGET_CLUSTER})')
plt.xlabel('Epochs')
plt.ylabel('Total MSE Loss')
plt.grid(True, alpha=0.3)
plt.savefig(f"{MODEL_DIR}/pretrain_loss_cluster_{TARGET_CLUSTER}.png")
print(f"학습 곡선 저장 완료: {MODEL_DIR}/pretrain_loss_cluster_{TARGET_CLUSTER}.png")