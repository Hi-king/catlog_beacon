import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import sys
import matplotlib.dates as mdates # datetime表示のためにインポート
import matplotlib.ticker as ticker # FuncFormatterのためにインポート
import japanize_matplotlib

# --- Parameters ---
MIN_DURATION_MINUTES = 20
min_duration_seconds = MIN_DURATION_MINUTES * 60
NOISE_SCALE = 30 # ノイズのスケール

def main(log_file_path):
    # 1. CSVデータ読み込み
    print(f"ログファイル: {log_file_path} を読み込みます。")

    df = pd.read_csv(log_file_path).rename(columns={"datetime": "timestamp", "predicted_locatio": "location", "predicted_location": "location"})

    # timestampをdatetimeに変換し、ソート
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 1. 同じ場所が連続する場合は最初のデータのみ残す
    df_filtered = df[df['location'].ne(df['location'].shift())].copy()

    # 「不明」な場所を除外
    df_filtered = df_filtered[df_filtered['location'] != '不明'].copy()

    # 2. 連続する移動間の時間差が閾値未満の点を削除
    if not df_filtered.empty: # フィルタリング後にデータが残っているか確認
        time_diff = df_filtered['timestamp'].diff().dt.total_seconds()
        # 最初の行(time_diff is NaT) または 時間差が閾値以上の行を保持
        df_filtered = df_filtered[(time_diff.isna()) | (time_diff >= min_duration_seconds)].copy()

    if df_filtered.empty:
        print(f"No significant stays longer than {MIN_DURATION_MINUTES} minutes found after filtering or all locations were '不明'. Exiting.")
        sys.exit() # exit() から sys.exit() に変更して終了コードを返す


    # --- Post-filtering Calculations ---
    # タイムスタンプを数値に変換（時間経過を数値で表現） - フィルタリング後に再計算
    # カラーバーの色分けに使用するため time_numeric を使用
    df_filtered['time_numeric'] = (df_filtered['timestamp'] - df_filtered['timestamp'].min()).dt.total_seconds()
    # 元のdfのtime_numericは不要になったため削除またはコメントアウト
    # df['time_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    # 滞在時間を計算（次の移動までの時間） - フィルタリング後に再計算
    df_filtered['stay_duration'] = df_filtered['timestamp'].shift(-1) - df_filtered['timestamp']
    df_filtered['stay_duration'] = df_filtered['stay_duration'].dt.total_seconds().fillna(0) # 最後の点は滞在時間0とする
    # print(df_filtered.stay_duration) # デバッグ用printはコメントアウト

    # --- 2. Room Coordinates ---
    room_positions = {
        'LIVING ROOM': (1350, 200),
        'リビング': (1350, 200),
        'DINING ROOM': (1350, 200),

        'KITCHEN': (1000, 800),
        'キッチン': (1000, 800),
        'OFFICE': (800, 200),
        '書斎': (800, 200),

        'BEDROOM': (180, 450),
        'ベッド下': (180, 450),
        'ベッド': (180, 450),

        'キャットタワー': (100, 320),

        'HALLWAY': (300, 800),
        '寝室前廊下': (300, 800),

        'クローゼット': (400, 400),

        '不明':(0, 0), # 不明は除外されるが、定義は残しておく
    }

    # locationをxy座標に変換 (df_filteredを使用)
    # フィルタリング後のデータフレームに対してのみ座標変換を実行
    df_filtered['x'] = df_filtered['location'].map(lambda loc: room_positions.get(loc, (0,0))[0])
    df_filtered['y'] = df_filtered['location'].map(lambda loc: room_positions.get(loc, (0,0))[1])

    # --- Noise Addition ---
    # 適度にノイズを付与して重なりを避ける (df_filteredを使用)
    def add_noise(x, y, scale=8):
        return x + np.random.uniform(-scale, scale), y + np.random.uniform(-scale, scale)

    noisy_coords = df_filtered.apply(lambda row: add_noise(row['x'], row['y'], scale=NOISE_SCALE), axis=1)
    df_filtered['x_noisy'] = noisy_coords.apply(lambda coord: coord[0])
    df_filtered['y_noisy'] = noisy_coords.apply(lambda coord: coord[1])

    # --- 3. Background Image ---
    # img = mpimg.imread('/mnt/data/348ff89b-4950-4517-81af-bb398574e73e.png')
    img_path = './adhoc/room.png'
    try:
        img = mpimg.imread(img_path)
    except FileNotFoundError:
        print(f"Error: Background image not found at {img_path}")
        # Create a dummy white background if image not found
        img = np.ones((600, 400, 3))


    import matplotlib.colors as mcolors

    # --- 4. Plotting ---
    # extentを画像のサイズに合わせる
    img_height, img_width, _ = img.shape
    # figsizeを画像のサイズに合わせて調整 (アスペクト比を維持しつつ、適宜調整)
    # 例: 画像の幅を基準にfigsizeの幅を決定し、高さはアスペクト比から計算
    base_width = 10 # 基準となるfigsizeの幅
    figsize_width = base_width
    figsize_height = base_width * (img_height / img_width)
    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
    ax.imshow(img, extent=(0, img_width, img_height, 0), aspect='auto')

    # 色を時間経過に合わせて変化させる (df_filteredを使用)
    # カラーバーは時間経過に対応させる
    norm = mcolors.Normalize(vmin=df_filtered['time_numeric'].min(), vmax=df_filtered['time_numeric'].max())
    cmap = plt.get_cmap('plasma')
    # カラーデータとして時間経過を使用
    colors = cmap(norm(df_filtered['time_numeric']))

    # 線を色付きで描画 (df_filteredを使用)
    # x軸とy軸は座標を使用
    for i in range(len(df_filtered)-1):
        ax.plot(df_filtered['x_noisy'].iloc[i:i+2], df_filtered['y_noisy'].iloc[i:i+2], color=colors[i], linewidth=2, alpha=0.7)

    # スキャッターで点を描く（滞在時間に応じて点の大きさを変更） (df_filteredを使用)
    # x軸とy軸は座標を使用, cに時間経過を使用
    size_scaling_factor = 0.05 # 点のサイズ調整用係数 (適宜調整)
    sc = ax.scatter(
        df_filtered['x_noisy'], # x軸は座標
        df_filtered['y_noisy'], # y軸は座標
        c=df_filtered['time_numeric'], # 色分けに時間経過を使用
        cmap='plasma',
        s=10 + df_filtered['stay_duration'] * size_scaling_factor, # stay_duration を使用
        alpha=0.8
    )

    # カラーバー追加 (df_filteredのtime_numericに基づく)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5) # shrinkでサイズ調整
    # カラーバーの表示形式を絶対時間（時:分:秒）に設定
    # time_numeric (経過秒数) を元のタイムスタンプの範囲にマッピングして表示
    start_time = df_filtered['timestamp'].min()
    end_time = df_filtered['timestamp'].max()
    # 経過秒数と絶対時間の対応を示すための Locator と Formatter を設定
    # ここでは、カラーバーのティック位置を時間経過（秒）で指定し、その位置に対応する絶対時間を表示する
    # 例: 0秒、60秒、120秒... の位置に、開始時刻から 0秒後、60秒後、120秒後... の絶対時間を表示

    # 経過秒数から絶対時間への変換関数
    def format_time_from_seconds(x, pos=None):
        if df_filtered.empty:
            return ""
        # 経過秒数 x を開始時刻に加算して絶対時間を計算
        absolute_time = start_time + pd.Timedelta(seconds=x)
        return absolute_time.strftime('%H:%M:%S')

    # ticker.FuncFormatter を使用
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_time_from_seconds))
    cbar.set_label('Time') # カラーバーラベルを絶対時間に合わせて変更

    # --- 5. Final Plot Settings ---
    # x軸とy軸の表示範囲を画像のサイズに合わせる
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)

    ax.set_title(f'猫がどこに滞在していたか ({MIN_DURATION_MINUTES}分以下の滞在は無視)')
    ax.axis('off') # x軸とy軸は座標なので非表示に戻す

    plt.tight_layout() # レイアウト調整
    plt.show()

if __name__ == "__main__":
    # argparseを使用してコマンドライン引数を処理
    parser = argparse.ArgumentParser(description='Plot cat walk from log file.')
    parser.add_argument('log_file', nargs='?', default='rssi_inference_log.csv',
                        help='Path to the log file (default: rssi_inference_log.csv)')
    args = parser.parse_args()

    main(args.log_file)
