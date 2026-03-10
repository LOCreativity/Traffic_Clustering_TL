import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# 환경 설정
DATA_DIR = './data/processed'
RESULT_DIR = './results'

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

print("역정규화 기반 군집별 실제 트래픽 시계열 흐름 분석 시작")

# 데이터 로드
X_tensor = torch.load(f"{DATA_DIR}/multitask_timeseries_top200.pt")
with open(f"{DATA_DIR}/cluster_assignments.pkl", 'rb') as f:
    cluster_assignments = pickle.load(f)
with open(f"{DATA_DIR}/metadata.pkl", 'rb') as f:
    metadata = pickle.load(f)

cell_ids = metadata['cell_ids']
scalers = metadata['scalers']
features = metadata['features']  # ['sms', 'call', 'internet']

num_clusters = len(set(cluster_assignments.values()))

# 군집별 실제 트래픽을 저장할 딕셔너리
cluster_actual_data = {i: [] for i in range(num_clusters)}

for idx, cell_id in enumerate(cell_ids):
    cluster_id = cluster_assignments[cell_id]
    scaler = scalers[cell_id]

    # 실제 볼륨(수치)으로 복원
    scaled_data = X_tensor[idx].numpy()
    actual_data = scaler.inverse_transform(scaled_data)

    cluster_actual_data[cluster_id].append(actual_data)

# 4개의 서브플롯으로 실제 트래픽 흐름 그리기 (sharey=True로 Y축 눈금 완전 통일)
fig, axes = plt.subplots(num_clusters, 1, figsize=(12, 3 * num_clusters), sharex=True, sharey=True)
colors = ['red', 'blue', 'green']

for i in range(num_clusters):
    ax = axes[i]
    c_data = np.array(cluster_actual_data[i])  # (격자수, 720, 3)
    c_mean = np.mean(c_data, axis=0)  # 군집 내 모든 격자의 평균 트래픽 흐름 계산

    for j, feature_name in enumerate(features):
        ax.plot(c_mean[:, j], label=feature_name.upper(), color=colors[j], alpha=0.8)

    ax.set_title(f'Cluster {i} Actual Traffic Flow (Cells: {len(c_data)})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Volume')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.xlabel('Time (Hours)', fontsize=12)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/actual_traffic_timeseries.png", dpi=300)
print(f"군집별 실제 트래픽 시계열 그래프 저장 완료: {RESULT_DIR}/actual_traffic_timeseries.png")