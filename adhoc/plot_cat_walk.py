import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

# 1. CSVデータ読み込み（サンプルデータを100点生成）
room_list = ['BEDROOM', 'HALLWAY', 'DINING ROOM', 'LIVING ROOM', 'KITCHEN', 'OFFICE', 'BATHROOM', 'LAUNDRY ROOM', '猫用トイレ', 'ソファー']

timestamps = pd.to_datetime('2025-04-28 21:00') + pd.to_timedelta(np.cumsum(np.random.exponential(scale=60, size=100)), unit='s')
locations = [random.choice(room_list) for _ in range(100)]




data = {
    'timestamp': timestamps,
    'location': locations
}

df = pd.DataFrame(data) # この行はサンプルデータ作成用なので、実際には不要かもしれません。コメントアウトまたは削除を検討。

# --- Parameters ---
MIN_DURATION_MINUTES = 5
min_duration_seconds = MIN_DURATION_MINUTES * 60

# --- Data Loading and Filtering ---
df = pd.read_csv("rssi_inference_log.csv").rename(columns={"datetime": "timestamp", "predicted_location": "location"})

# timestampをdatetimeに変換し、ソート
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# 1. 同じ場所が連続する場合は最初のデータのみ残す
df_filtered = df[df['location'].ne(df['location'].shift())].copy()

# 2. 連続する移動間の時間差が閾値未満の点を削除
if not df_filtered.empty: # フィルタリング後にデータが残っているか確認
    time_diff = df_filtered['timestamp'].diff().dt.total_seconds()
    # 最初の行(time_diff is NaT) または 時間差が閾値以上の行を保持
    df_filtered = df_filtered[(time_diff.isna()) | (time_diff >= min_duration_seconds)].copy()

if df_filtered.empty:
    print(f"No significant stays longer than {MIN_DURATION_MINUTES} minutes found after filtering. Exiting.")
    exit()


# --- Post-filtering Calculations ---
# タイムスタンプを数値に変換（時間経過を数値で表現） - フィルタリング後に再計算
df_filtered['time_numeric'] = (df_filtered['timestamp'] - df_filtered['timestamp'].min()).dt.total_seconds()
df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

# 滞在時間を計算（次の移動までの時間） - フィルタリング後に再計算
df_filtered['stay_duration'] = df_filtered['timestamp'].shift(-1) - df_filtered['timestamp']
df_filtered['stay_duration'] = df_filtered['stay_duration'].dt.total_seconds().fillna(0) # 最後の点は滞在時間0とする
# print(df_filtered.stay_duration) # デバッグ用printはコメントアウト

# --- 2. Room Coordinates ---
room_positions = {
    'LIVING ROOM': (100, 150),
    'リビング': (100, 150),
    'DINING ROOM': (200, 150),
    'KITCHEN': (300, 150),
    'キッチン': (300, 150),
    'OFFICE': (100, 300),
    '書斎': (100, 300),
    'BATHROOM': (200, 300),
    'LAUNDRY ROOM': (300, 300),
    'BEDROOM': (200, 450),
    'ベッド下': (200, 450),
    'キャットタワー': (200, 450),
    'HALLWAY': (300, 450),
    '寝室前廊下': (300, 450),
    '猫用トイレ': (200, 100),
    'ソファー': (100, 100),
    '不明':(0, 0),
}

# locationをxy座標に変換 (df_filteredを使用)
for location in df_filtered['location'].unique():
     if location not in room_positions:
        print(f"Warning: Location '{location}' not found in room_positions. Assigning (0,0).")
df_filtered['x'] = df_filtered['location'].map(lambda loc: room_positions.get(loc, (0,0))[0])
df_filtered['y'] = df_filtered['location'].map(lambda loc: room_positions.get(loc, (0,0))[1])

# --- Noise Addition ---
# 適度にノイズを付与して重なりを避ける (df_filteredを使用)
def add_noise(x, y, scale=8):
    return x + np.random.uniform(-scale, scale), y + np.random.uniform(-scale, scale)

noisy_coords = df_filtered.apply(lambda row: add_noise(row['x'], row['y']), axis=1)
df_filtered['x_noisy'] = noisy_coords.apply(lambda coord: coord[0])
df_filtered['y_noisy'] = noisy_coords.apply(lambda coord: coord[1])

# --- 3. Background Image ---
# img = mpimg.imread('/mnt/data/348ff89b-4950-4517-81af-bb398574e73e.png')
img_path = '/Users/keisuke.ogaki/Downloads/Screenshot_20250428-211739.png'
try:
    img = mpimg.imread(img_path)
except FileNotFoundError:
    print(f"Error: Background image not found at {img_path}")
    # Create a dummy white background if image not found
    img = np.ones((600, 400, 3))


import matplotlib.colors as mcolors

# --- 4. Plotting ---
fig, ax = plt.subplots(figsize=(12, 18))
# extentをタプルに変更, aspectを追加
ax.imshow(img, extent=(0, 400, 600, 0), aspect='auto')

# 色を時間に合わせて変化させる (df_filteredを使用)
norm = mcolors.Normalize(vmin=df_filtered['time_numeric'].min(), vmax=df_filtered['time_numeric'].max())
cmap = plt.get_cmap('plasma')
colors = cmap(norm(df_filtered['time_numeric']))

# 線を色付きで描画 (df_filteredを使用)
for i in range(len(df_filtered)-1):
    ax.plot(df_filtered['x_noisy'].iloc[i:i+2], df_filtered['y_noisy'].iloc[i:i+2], color=colors[i], linewidth=2, alpha=0.7)

# スキャッターで点を描く（滞在時間に応じて点の大きさを変更） (df_filteredを使用)
size_scaling_factor = 0.05 # 点のサイズ調整用係数 (適宜調整)
sc = ax.scatter(
    df_filtered['x_noisy'],
    df_filtered['y_noisy'],
    c=df_filtered['time_numeric'],
    cmap='plasma',
    s=10 + df_filtered['stay_duration'] * size_scaling_factor, # stay_duration を使用
    alpha=0.8
)

# カラーバー追加 (df_filteredのtime_numericに基づく)
cbar = plt.colorbar(sc, ax=ax, shrink=0.5) # shrinkでサイズ調整
cbar.set_label(f'Time elapsed since first significant stay (seconds)')

# --- 5. Final Plot Settings ---
ax.set_xlim(0, 400)
ax.set_ylim(600, 0)
ax.set_title(f'Cat Walk (Stays >= {MIN_DURATION_MINUTES} min marked)')
ax.axis('off')
plt.show()
