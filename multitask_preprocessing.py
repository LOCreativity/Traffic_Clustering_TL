import pandas as pd
import numpy as np
import torch
import glob
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

# 환경 설정 및 디렉토리 준비
DATA_PATTERN = './data/raw/*_hourly_formatted.csv'
SAVE_DIR = './data/processed'
TOP_N_CELLS = 200 # 군집화에 사용할 유의미한 상위 격자 개수

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print("다중 서비스(SMS, Call, Internet) 트래픽 데이터 통합 중...")

# 데이터 병합 및 Multi-Task 특성 생성
file_list = sorted(glob.glob(DATA_PATTERN))
if not file_list:
    print(f"[오류] {DATA_PATTERN} 경로에 파일이 없음. raw 폴더에 파일을 복사했는지 확인.")
    exit()

all_df = []
for file in file_list:
    # low_memory=False로 대용량 메모리 경고 방지
    temp_df = pd.read_csv(file, parse_dates=['datetime'], low_memory=False)

    # SMS와 Call은 in/out을 합산하여 하나의 서비스 지표로 통합
    temp_df['sms'] = temp_df['smsin'].fillna(0) + temp_df['smsout'].fillna(0)
    temp_df['call'] = temp_df['callin'].fillna(0) + temp_df['callout'].fillna(0)
    temp_df['internet'] = temp_df['internet'].fillna(0)

    # 시간 및 격자별로 3가지 트래픽 합산
    agg_df = temp_df.groupby(['datetime', 'CellID'])[['sms', 'call', 'internet']].sum().reset_index()
    all_df.append(agg_df)

df_total = pd.concat(all_df, axis=0).reset_index(drop=True)
print(f"통합 완료. 총 데이터 행 수: {len(df_total)}")

# 유의미한 타겟 격자(Top N) 추출
# 인터넷 트래픽 총합이 가장 높은 상위 N개 격자 선정
cell_totals = df_total.groupby('CellID')['internet'].sum().sort_values(ascending=False)
target_cells = cell_totals.head(TOP_N_CELLS).index.tolist()
print(f"상위 {TOP_N_CELLS}개 격자 추출 완료.")

df_top_cells = df_total[df_total['CellID'].isin(target_cells)].copy()

# 시계열 연속성 보장 및 피벗
# 모든 격자가 동일한 타임라인을 갖도록 Pivot 적용
pivot_df = df_top_cells.pivot_table(
    index='datetime',
    columns='CellID',
    values=['sms', 'call', 'internet'],
    aggfunc='sum'
).fillna(0)

# 1시간 단위 리샘플링 (누락된 중간 시간대는 0으로 채움)
pivot_df = pivot_df.resample('h').sum().fillna(0)
total_hours = len(pivot_df)
print(f"연속된 총 시간(Hours): {total_hours}시간")

# 3차원 텐서 변환 및 독립적 정규화
# 최종 데이터 형태: (격자 수, 시간, 특성 수) -> (200, total_hours, 3)
num_cells = len(target_cells)
num_features = 3 # sms, call, internet

tensor_data = np.zeros((num_cells, total_hours, num_features))
scalers = {}

for idx, cell_id in enumerate(target_cells):
    # 해당 격자의 3가지 트래픽 데이터 추출 (멀티인덱스 접근)
    cell_data = pivot_df.xs(cell_id, level='CellID', axis=1)[['sms', 'call', 'internet']].values

    # 격자별 독립적인 정규화 (패턴의 모양에 집중하기 위함)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(cell_data)

    tensor_data[idx, :, :] = scaled_data
    scalers[cell_id] = scaler # 추후 평가 시 역정규화를 위해 저장

X_tensor = torch.FloatTensor(tensor_data)

# 결과물 저장
# 텐서 저장
torch.save(X_tensor, f"{SAVE_DIR}/multitask_timeseries_top{TOP_N_CELLS}.pt")

# 메타데이터 저장 (Cell ID 매핑 및 스케일러 객체)
metadata = {
    'cell_ids': target_cells,
    'scalers': scalers,
    'features': ['sms', 'call', 'internet']
}
with open(f"{SAVE_DIR}/metadata.pkl", 'wb') as f:
    pickle.dump(metadata, f)

print("\n=======================================")
print(f"다중 작업 전처리 완료")
print(f"- 생성된 텐서 형태: {X_tensor.shape} (격자 수, 시간, 트래픽 종류)")
print(f"- 데이터 저장 위치: {SAVE_DIR}/")
print("=======================================")