import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# 환경 설정
DATA_DIR = './data/processed'
MODEL_DIR = './models'
RESULT_DIR = './results'

PRETRAINED_MODEL_PATH = f"{MODEL_DIR}/pretrained_cluster_0.pth"
TARGET_CLUSTER = 0 # 전이 학습을 테스트할 이질적인 군집

WINDOW_SIZE = 168
PREDICT_SIZE = 24
FINETUNE_EPOCHS = 500
PATIENCE = 50
LEARNING_RATE = 0.0001 # 학습률을 낮춰 사전 학습 지식이 크게 망가지는 것을 방지

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 타겟 격자 데이터 준비
print(f"Cluster {TARGET_CLUSTER}의 격자를 대상으로 전이 학습 준비 중")


X_tensor = torch.load(f"{DATA_DIR}/multitask_timeseries_top200.pt")
with open(f"{DATA_DIR}/cluster_assignments.pkl", 'rb') as f:
    cluster_assignments = pickle.load(f)
with open(f"{DATA_DIR}/metadata.pkl", 'rb') as f:
    metadata = pickle.load(f)

cell_ids = metadata['cell_ids']
scalers = metadata['scalers']
features = metadata['features']

# Cluster X에 속한 격자 중 첫 번째 격자를 테스트 타겟으로 선정
target_indices = [idx for idx, cell_id in enumerate(cell_ids) if cluster_assignments[cell_id] == TARGET_CLUSTER]
if not target_indices:
    print(f"오류: Cluster {TARGET_CLUSTER}에 할당된 격자가 없음.")
    exit()

target_idx = target_indices[0]
target_cell_id = cell_ids[target_idx]
target_scaler = scalers[target_cell_id]

print(f"타겟 격자 선정 완료: Cell ID {target_cell_id} (Cluster {TARGET_CLUSTER})")

# 슬라이딩 윈도우 생성
def create_multistep_sequences(data, window, predict):
    xs, ys = [], []
    for i in range(len(data) - window - predict + 1):
        xs.append(data[i : (i + window)])
        ys.append(data[(i + window) : (i + window + predict)])
    return np.array(xs), np.array(ys)

cell_data = X_tensor[target_idx].numpy()
X_seq, y_seq = create_multistep_sequences(cell_data, WINDOW_SIZE, PREDICT_SIZE)

X_seq = torch.FloatTensor(X_seq)
y_seq = torch.FloatTensor(y_seq)

# 데이터 분할 (전체 샘플의 80%는 파인튜닝, 20%는 테스트용으로 동적 분할)
total_samples = len(X_seq)
train_size = int(total_samples * 0.8)

train_X, test_X = X_seq[:train_size], X_seq[train_size:]
train_y, test_y = y_seq[:train_size], y_seq[train_size:]

print(f"총 시퀀스 샘플: {total_samples}개")
print(f"파인튜닝(Train) 사용: {train_size}개 | 테스트(Test) 사용: {total_samples - train_size}개")

train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=1, shuffle=False)

# 모델 정의 및 사전 학습 가중치 로드
class MultiTaskTrafficLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=3, output_size=24):
        super(MultiTaskTrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_sms = nn.Linear(hidden_size, output_size)
        self.fc_call = nn.Linear(hidden_size, output_size)
        self.fc_internet = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc_sms(last_hidden), self.fc_call(last_hidden), self.fc_internet(last_hidden)

model = MultiTaskTrafficLSTM().to(device)
model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
criterion = nn.MSELoss()

# Zero-shot 테스트 (파인튜닝 전)
print("\nZero-shot 테스트 (사전 학습 모델 그대로 적용)")
model.eval()
zero_shot_loss = 0
zero_shot_preds = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        p_sms, p_call, p_int = model(data)

        # 3차원으로 합치기: (Batch, 24, 3)
        preds = torch.stack((p_sms, p_call, p_int), dim=-1)
        zero_shot_preds.append(preds.cpu().numpy())

        loss = criterion(p_sms, target[:, :, 0]) + criterion(p_call, target[:, :, 1]) + criterion(p_int, target[:, :, 2])
        zero_shot_loss += loss.item()

print(f"Zero-shot Total Loss: {zero_shot_loss / len(test_loader):.6f}")

# 파인튜닝
print(f"\n파인튜닝 시작 {FINETUNE_EPOCHS} Epochs")

for param in model.lstm.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(FINETUNE_EPOCHS):
    model.train()
    epoch_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        p_sms, p_call, p_int = model(data)

        loss = criterion(p_sms, target[:, :, 0]) + criterion(p_call, target[:, :, 1]) + criterion(p_int, target[:, :, 2])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            p_sms, p_call, p_int = model(data)
            v_loss = criterion(p_sms, target[:, :, 0]) + criterion(p_call, target[:, :, 1]) + criterion(p_int,                                                                                 target[:, :, 2])
            val_loss += v_loss.item()

    avg_val_loss = val_loss / len(test_loader)

    # 조기 종료를 Validation Loss 기준으로 작동
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        # 가장 성능이 좋은 시점의 모델 가중치 임시 저장
        torch.save(model.state_dict(), f"{MODEL_DIR}/best_finetuned.pth")
    else:
        early_stop_counter += 1

    if (epoch+1) % 10 == 0:
        print(f"Fine-tune Epoch [{epoch+1}/{FINETUNE_EPOCHS}] Train Loss: {avg_train_loss:.6f}")

    if early_stop_counter >= PATIENCE:
        print(f"파인튜닝 조기 종료. (Epoch {epoch+1})")
        break

# 가장 성능이 좋았던 모델 상태로 복구
model.load_state_dict(torch.load(f"{MODEL_DIR}/best_finetuned.pth", weights_only=True))

# 파인튜닝 후 테스트 및 결과 시각화
print("\n파인튜닝 후 최종 테스트 및 시각화")
model.eval()
finetuned_loss = 0
finetuned_preds = []
actuals = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        p_sms, p_call, p_int = model(data)

        preds = torch.stack((p_sms, p_call, p_int), dim=-1)
        finetuned_preds.append(preds.cpu().numpy())
        actuals.append(target.cpu().numpy())

        loss = criterion(p_sms, target[:, :, 0]) + criterion(p_call, target[:, :, 1]) + criterion(p_int, target[:, :, 2])
        finetuned_loss += loss.item()

final_loss = finetuned_loss / len(test_loader)
print(f"Fine-tuned Total Loss: {final_loss:.6f}")

# Zero-shot loss가 0인 경우를 방지
if zero_shot_loss > 0:
    improvement = ((zero_shot_loss / len(test_loader) - final_loss) / (zero_shot_loss / len(test_loader))) * 100
    print(f"성능 향상률: {improvement:.2f}% 개선됨")
else:
    print("Zero_shot loss가 너무 많아 향상률을 개선할 수 없음.")

# 마지막 샘플 하나를 뽑아서 실제 트래픽 수치로 역정규화
sample_idx = -1
actual_scaled = actuals[sample_idx][0]
zero_scaled = zero_shot_preds[sample_idx][0]
fine_scaled = finetuned_preds[sample_idx][0]

actual_inv = target_scaler.inverse_transform(actual_scaled)
zero_inv = target_scaler.inverse_transform(zero_scaled)
fine_inv = target_scaler.inverse_transform(fine_scaled)

# 시각화 (SMS, Call, Internet 3개의 Subplot)
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
colors = ['red', 'blue', 'green']

for i in range(3):
    ax = axes[i]
    ax.plot(actual_inv[:, i], label='Actual Target', marker='o', color='black', linewidth=2)
    ax.plot(zero_inv[:, i], label='Zero-shot (Before FT)', marker='^', linestyle=':', color='gray', alpha=0.7)
    ax.plot(fine_inv[:, i], label='Fine-tuned (After FT)', marker='x', linestyle='--', color=colors[i], linewidth=2)

    ax.set_title(f'{features[i].upper()} Traffic Prediction', fontsize=12)
    ax.set_ylabel('Traffic Volume')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.xlabel('Hours Ahead', fontsize=12)
plt.suptitle(f'Generalization Test on Cell {target_cell_id} (Cluster {TARGET_CLUSTER})', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/transfer_learning_result_cell_{target_cell_id}.png", dpi=300)

print(f"\n최종 결과 그래프 저장 완료: {RESULT_DIR}/transfer_learning_result_cell_{target_cell_id}.png")

print("\n[가장 예측이 빗나간 구간 Top 5 분석]")

# SMS, Call, Internet의 절대 오차를 합산
diff_sms = np.abs(actual_inv[:, 0] - fine_inv[:, 0])
diff_call = np.abs(actual_inv[:, 1] - fine_inv[:, 1])
diff_int = np.abs(actual_inv[:, 2] - fine_inv[:, 2])
total_diff = diff_sms + diff_call + diff_int

# 오차가 가장 큰 상위 5개 시간대(인덱스) 추출
worst_indices = total_diff.argsort()[-5:][::-1]

for rank, idx in enumerate(worst_indices):
    print(f"--- 랭킹 {rank+1}위 (미래 {idx}시간 후) ---")
    print(f"  [SMS] 실제: {actual_inv[idx, 0]:.4f}  | 예측: {fine_inv[idx, 0]:.4f}  | 차이: {diff_sms[idx]:.4f}")
    print(f"  [CALL] 실제: {actual_inv[idx, 1]:.4f} | 예측: {fine_inv[idx, 1]:.4f}  | 차이: {diff_call[idx]:.4f}")
    print(f"  [INT] 실제: {actual_inv[idx, 2]:.4f}  | 예측: {fine_inv[idx, 2]:.4f}  | 차이: {diff_int[idx]:.4f}")