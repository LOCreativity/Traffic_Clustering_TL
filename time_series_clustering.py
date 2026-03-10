import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans

# 환경 설정 및 데이터 로드
DATA_DIR = './data/processed'
RESULT_DIR = './results'

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

print("전처리된 텐서 데이터 및 메타데이터 로드 중")
X_tensor = torch.load(f"{DATA_DIR}/multitask_timeseries_top200.pt")
X_data = X_tensor.numpy() # (200, 720, 3)

with open(f"{DATA_DIR}/metadata.pkl", 'rb') as f:
    metadata = pickle.load(f)

cell_ids = metadata['cell_ids']
features = metadata['features'] # ['sms', 'call', 'internet']

# 최적의 군집 수(K) 자동 탐색 (Elbow Method)
print("\n최적의 군집 수(K) 자동 탐색 시작")
print("K를 2부터 10까지 변경하며 DTW 오차를 계산")

K_range = range(2, 11)
inertias = []

for k in K_range:
    # 빠른 탐색을 위해 max_iter를 약간 줄여서 테스트
    km_test = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=5, random_state=42, n_jobs=-1)
    km_test.fit(X_data)
    inertias.append(km_test.inertia_)
    print(f"K={k} 일 때 오차: {km_test.inertia_:.2f}")

# 기하학적 거리를 이용한 Elbow 지점 자동 계산 로직
# 시작점(K=2)과 끝점(K=10)을 잇는 직선 방정식에서 가장 멀리 떨어진 점을 찾음
p1 = np.array([K_range[0], inertias[0]])
p2 = np.array([K_range[-1], inertias[-1]])
distances = []

for i in range(len(K_range)):
    p3 = np.array([K_range[i], inertias[i]])
    # 점과 선 사이의 거리 공식 (Cross product 이용)
    distance = np.abs(np.cross(p2 - p1, p3 - p1)) / np.linalg.norm(p2 - p1)
    distances.append(distance)

# 거리가 가장 먼 지점이 바로 최적의 군집 수(Elbow Point)
optimal_idx = np.argmax(distances)
OPTIMAL_K = K_range[optimal_idx]

print(f"\n데이터 분석 결과 최적의 군집 수(K): {OPTIMAL_K}개")

# 엘보우 기법 결과 그래프 저장
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o', linestyle='-', color='b')
plt.axvline(x=OPTIMAL_K, color='r', linestyle='--', label=f'Optimal K = {OPTIMAL_K}')
plt.plot(K_range[optimal_idx], inertias[optimal_idx], 'ro', markersize=8)
plt.title('Elbow Method for Optimal K (DTW KMeans)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/dtw_elbow_method.png", dpi=300)
print(f"엘보우 기법 분석 그래프 저장 완료: {RESULT_DIR}/dtw_elbow_method.png")

# 도출된 최적의 K로 최종 군집화 수행
print(f"\n최적 군집 수({OPTIMAL_K})를 바탕으로 최종 DTW K-Means 군집화 시작")
km_final = TimeSeriesKMeans(
    n_clusters=OPTIMAL_K,
    metric="dtw",
    max_iter=10,
    random_state=42,
    n_jobs=-1
)

labels = km_final.fit_predict(X_data)
print("최종 군집화 완료")

unique_labels, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count}개 격자 할당")

cluster_assignments = {cell_id: int(label) for cell_id, label in zip(cell_ids, labels)}

with open(f"{DATA_DIR}/cluster_assignments.pkl", 'wb') as f:
    pickle.dump(cluster_assignments, f)
print(f"군집 할당 결과 저장 완료: {DATA_DIR}/cluster_assignments.pkl")

# 군집별 대표 패턴(Centroid) 시각화
centroids = km_final.cluster_centers_

fig, axes = plt.subplots(OPTIMAL_K, 1, figsize=(12, 3 * OPTIMAL_K), sharex=True)
if OPTIMAL_K == 1:
    axes = [axes]

colors = ['red', 'blue', 'green']

for i in range(OPTIMAL_K):
    ax = axes[i]
    for j, feature_name in enumerate(features):
        ax.plot(centroids[i, :, j], label=feature_name.upper(), color=colors[j], alpha=0.8)

    ax.set_title(f'Cluster {i} Representative Pattern (Cells: {counts[i]})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Traffic')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.xlabel('Time (Hours)', fontsize=12)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/dtw_cluster_centroids.png", dpi=300)
print(f"최종 군집별 대표 패턴 그래프 저장 완료: {RESULT_DIR}/dtw_cluster_centroids.png")