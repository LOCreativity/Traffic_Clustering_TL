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
RESULT_DIR = './results/cross_transfer'

WINDOW_SIZE = 168
PREDICT_SIZE = 24
FINETUNE_EPOCHS = 300
PATIENCE = 30
LEARNING_RATE = 0.0001
NUM_CLUSTERS = 4

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")

# 공통 데이터 로드
print("데이터 및 메타데이터 로드 중...")
X_tensor = torch.load(f"{DATA_DIR}/multitask_timeseries_top200.pt")
with open(f"{DATA_DIR}/cluster_assignments.pkl", 'rb') as f:
    cluster_assignments = pickle.load(f)
with open(f"{DATA_DIR}/metadata.pkl", 'rb') as f:
    metadata = pickle.load(f)

cell_ids = metadata['cell_ids']
scalers = metadata['scalers']
features = metadata['features']


def create_multistep_sequences(data, window, predict):
    xs, ys = [], []
    for i in range(len(data) - window - predict + 1):
        xs.append(data[i: (i + window)])
        ys.append(data[(i + window): (i + window + predict)])
    return np.array(xs), np.array(ys)


# 다중 작업 LSTM 모델 정의
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

# 4x4 교차 전이 학습 수행 (이중 루프)
results_matrix = np.zeros((NUM_CLUSTERS, NUM_CLUSTERS)) # 결과 요약을 저장할 딕셔너리

for source_cluster in range(NUM_CLUSTERS):
    pretrained_path = f"{MODEL_DIR}/pretrained_cluster_{source_cluster}.pth"

    if not os.path.exists(pretrained_path):
        print(f"\nSource Cluster {source_cluster}의 사전 학습 모델이 없어 건너뜀: {pretrained_path}")
        continue

    for target_cluster in range(NUM_CLUSTERS):
        print(f"\n{'=' * 50}")
        print(f"실험: Source Cluster {source_cluster} -> Target Cluster {target_cluster}")
        print(f"{'=' * 50}")

        # 타겟 격자 선정 (해당 군집의 첫 번째 격자를 대표로 사용)
        target_indices = [idx for idx, cell_id in enumerate(cell_ids) if cluster_assignments[cell_id] == target_cluster]
        if not target_indices:
            continue

        target_idx = target_indices[0]
        target_cell_id = cell_ids[target_idx]
        target_scaler = scalers[target_cell_id]

        # 시퀀스 분할 (80:20)
        cell_data = X_tensor[target_idx].numpy()
        X_seq, y_seq = create_multistep_sequences(cell_data, WINDOW_SIZE, PREDICT_SIZE)
        X_seq, y_seq = torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)

        total_samples = len(X_seq)
        train_size = int(total_samples * 0.8)

        train_loader = DataLoader(TensorDataset(X_seq[:train_size], y_seq[:train_size]), batch_size=16, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_seq[train_size:], y_seq[train_size:]), batch_size=1, shuffle=False)

        # 모델 초기화 및 가중치 로드 (매 반복마다 새롭게 덮어씌움)
        model = MultiTaskTrafficLSTM().to(device)
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
        criterion = nn.MSELoss()

        # Zero-shot 테스트
        model.eval()
        zero_shot_loss = 0
        zero_shot_preds, actuals = [], []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                p_sms, p_call, p_int = model(data)

                preds = torch.stack((p_sms, p_call, p_int), dim=-1)
                zero_shot_preds.append(preds.cpu().numpy())
                actuals.append(target.cpu().numpy())

                loss = criterion(p_sms, target[:, :, 0]) + criterion(p_call, target[:, :, 1]) + criterion(p_int,target[:, :, 2])
                zero_shot_loss += loss.item()

        # 파인튜닝
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(FINETUNE_EPOCHS):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                p_sms, p_call, p_int = model(data)
                loss = criterion(p_sms, target[:, :, 0]) + criterion(p_call, target[:, :, 1]) + criterion(p_int,
                                                                                                          target[:, :,
                                                                                                          2])
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    p_sms, p_call, p_int = model(data)
                    val_loss += (criterion(p_sms, target[:, :, 0]) + criterion(p_call, target[:, :, 1]) + criterion(
                        p_int, target[:, :, 2])).item()

            avg_val_loss = val_loss / len(test_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), f"{MODEL_DIR}/temp_finetuned.pth")
            else:
                early_stop_counter += 1

            if early_stop_counter >= PATIENCE:
                break

        # 파인튜닝 후 최종 테스트
        model.load_state_dict(torch.load(f"{MODEL_DIR}/temp_finetuned.pth", weights_only=True))
        model.eval()
        finetuned_loss = 0
        finetuned_preds = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                p_sms, p_call, p_int = model(data)
                preds = torch.stack((p_sms, p_call, p_int), dim=-1)
                finetuned_preds.append(preds.cpu().numpy())
                finetuned_loss += (criterion(p_sms, target[:,:,0]) + criterion(p_call, target[:,:,1]) + criterion(p_int, target[:,:,2])).item()

        final_loss = finetuned_loss / len(test_loader)
        improvement = ((zero_shot_loss / len(test_loader) - final_loss) / (zero_shot_loss / len(test_loader))) * 100
        results_matrix[source_cluster, target_cluster] = improvement

        print(f"Fine-tuned Loss: {final_loss:.6f} (성능 향상률: {improvement:.2f}%)")

        # 그래프 시각화 및 개별 파일 저장
        sample_idx = -1
        actual_scaled = actuals[sample_idx][0]
        zero_scaled = zero_shot_preds[sample_idx][0]
        fine_scaled = finetuned_preds[sample_idx][0]

        actual_inv = target_scaler.inverse_transform(actual_scaled)
        zero_inv = target_scaler.inverse_transform(zero_scaled)
        fine_inv = target_scaler.inverse_transform(fine_scaled)

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
        # 그래프 제목에 Source와 Target 군집 명시
        plt.suptitle(f'Transfer: Src Cluster {source_cluster} -> Tgt Cluster {target_cluster} (Cell {target_cell_id})', fontsize=16, y=0.98)
        plt.tight_layout()

        filename = f"{RESULT_DIR}/Src_{source_cluster}_to_Tgt_{target_cluster}_cell_{target_cell_id}.png"
        plt.savefig(filename, dpi=300)
        plt.close(fig)  # 메모리 관리를 위해 그래프 닫기

print("\n" + "="*50)
print("모든 16가지 조합의 전이 학습 및 그래프 생성 완료")
print(f"결과 그래프 저장 폴더: {RESULT_DIR}/")
print("="*50)

# 최종 4x4 개선율 매트릭스 출력
print("\n교차 전이 학습 성능 향상률(%) 매트릭스")
print("행(Row): Source 모델 / 열(Col): Target 군집")
print("          Tgt 0     Tgt 1     Tgt 2     Tgt 3")
for i in range(NUM_CLUSTERS):
    row_str = f"Src {i} |"
    for j in range(NUM_CLUSTERS):
        row_str += f"{results_matrix[i, j]:8.2f}% "
    print(row_str)