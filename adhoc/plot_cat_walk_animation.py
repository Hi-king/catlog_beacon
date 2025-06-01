import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import numpy as np
import sys
import matplotlib.ticker as ticker
import japanize_matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- Parameters ---
MIN_DURATION_MINUTES = 20
min_duration_seconds = MIN_DURATION_MINUTES * 60
NOISE_SCALE = 30 # ノイズのスケール
DEFAULT_FPS = 10  # デフォルトのアニメーションフレームレート
DEFAULT_INTERVAL = 100  # デフォルトのミリ秒単位のフレーム間隔

def main(log_file_path, output_format=None, output_file=None, speed_factor=1.0, fps=None):
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
    if not df_filtered.empty:
        time_diff = df_filtered['timestamp'].diff().dt.total_seconds()
        df_filtered = df_filtered[(time_diff.isna()) | (time_diff >= min_duration_seconds)].copy()

    if df_filtered.empty:
        print(f"No significant stays longer than {MIN_DURATION_MINUTES} minutes found after filtering or all locations were '不明'. Exiting.")
        sys.exit()

    # --- Post-filtering Calculations ---
    df_filtered['time_numeric'] = (df_filtered['timestamp'] - df_filtered['timestamp'].min()).dt.total_seconds()
    df_filtered['stay_duration'] = df_filtered['timestamp'].shift(-1) - df_filtered['timestamp']
    df_filtered['stay_duration'] = df_filtered['stay_duration'].dt.total_seconds().fillna(0)

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
        '不明':(0, 0),
    }

    # locationをxy座標に変換
    df_filtered['x'] = df_filtered['location'].map(lambda loc: room_positions.get(loc, (0,0))[0])
    df_filtered['y'] = df_filtered['location'].map(lambda loc: room_positions.get(loc, (0,0))[1])

    # --- Noise Addition ---
    def add_noise(x, y, scale=8):
        return x + np.random.uniform(-scale, scale), y + np.random.uniform(-scale, scale)

    noisy_coords = df_filtered.apply(lambda row: add_noise(row['x'], row['y'], scale=NOISE_SCALE), axis=1)
    df_filtered['x_noisy'] = noisy_coords.apply(lambda coord: coord[0])
    df_filtered['y_noisy'] = noisy_coords.apply(lambda coord: coord[1])

    # --- 3. Background Image ---
    img_path = './adhoc/room.png'
    try:
        img = mpimg.imread(img_path)
    except FileNotFoundError:
        print(f"Error: Background image not found at {img_path}")
        img = np.ones((600, 400, 3))

    # --- 4. Cat Image ---
    cat_img_path = pathlib.Path(__file__).parent / 'cat.png'
    try:
        cat_img = mpimg.imread(cat_img_path)
        # 猫画像のサイズを調整（必要に応じて）
        cat_zoom = 0.1  # 猫画像のスケール
    except FileNotFoundError:
        print(f"Error: Cat image not found at {cat_img_path}")
        # 猫画像がない場合は、単純な円で代替
        cat_img = None
        cat_zoom = 0.1  # デフォルト値を設定

    # --- 5. Animation Setup ---
    img_height, img_width, _ = img.shape
    base_width = 10
    figsize_width = base_width
    figsize_height = base_width * (img_height / img_width)
    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
    
    # 背景画像を表示
    ax.imshow(img, extent=(0, img_width, img_height, 0), aspect='auto')
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.axis('off')
    
    # 軌跡を描画するための線オブジェクト
    line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5)
    
    # 猫の位置を表示するオブジェクト
    cat_annotation = None
    cat_point = None
    if cat_img is not None:
        # 猫画像を使用
        imagebox = OffsetImage(cat_img, zoom=cat_zoom)
        cat_annotation = AnnotationBbox(imagebox, (0, 0), frameon=False)
        ax.add_artist(cat_annotation)
    else:
        # 円で代替
        cat_point, = ax.plot([], [], 'ro', markersize=20)
    
    # 時刻表示用のテキスト
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=18, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 場所表示用のテキスト
    location_text = ax.text(0.02, 0.89, '', transform=ax.transAxes, 
                           fontsize=16, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # タイトル
    ax.set_title(f'信号強度から推定した猫の移動経路')
    
    # 移動経路を補間する関数
    def interpolate_path(start_x, start_y, end_x, end_y, num_frames):
        """2点間を補間して滑らかな移動経路を生成"""
        x_path = np.linspace(start_x, end_x, num_frames)
        y_path = np.linspace(start_y, end_y, num_frames)
        return x_path, y_path
    
    # アニメーション用のフレームデータを準備
    frames_data = []
    for i in range(len(df_filtered)):
        if i < len(df_filtered) - 1:
            # 現在の位置から次の位置への移動を補間
            start_x = df_filtered['x_noisy'].iloc[i]
            start_y = df_filtered['y_noisy'].iloc[i]
            end_x = df_filtered['x_noisy'].iloc[i + 1]
            end_y = df_filtered['y_noisy'].iloc[i + 1]
            
            # 滞在時間に応じてフレーム数を決定（速度係数で調整）
            # speed_factor > 1 で高速化（フレーム数減少）、< 1 で低速化（フレーム数増加）
            base_stay_frames = min(max(int(df_filtered['stay_duration'].iloc[i] / 10), 10), 100)
            stay_frames = max(int(base_stay_frames / speed_factor), 1)
            
            # 現在位置での滞在フレーム（時間を進める）
            stay_duration_seconds = df_filtered['stay_duration'].iloc[i]
            for frame_idx in range(stay_frames):
                # 滞在中の経過時間を計算
                time_progress = (frame_idx / stay_frames) * stay_duration_seconds
                current_timestamp = df_filtered['timestamp'].iloc[i] + pd.Timedelta(seconds=time_progress)
                
                frames_data.append({
                    'x': start_x,
                    'y': start_y,
                    'timestamp': current_timestamp,
                    'location': df_filtered['location'].iloc[i],
                    'trail_x': df_filtered['x_noisy'].iloc[:i+1].tolist(),
                    'trail_y': df_filtered['y_noisy'].iloc[:i+1].tolist()
                })
            
            # 移動フレーム（速度係数で調整）
            base_move_frames = 20  # 基本の移動フレーム数
            move_frames = max(int(base_move_frames / speed_factor), 1)
            x_path, y_path = interpolate_path(start_x, start_y, end_x, end_y, move_frames)
            # 移動時間を仮定（次の位置までの時間の10%とする）
            if i + 1 < len(df_filtered):
                move_duration = (df_filtered['timestamp'].iloc[i+1] - df_filtered['timestamp'].iloc[i]).total_seconds() * 0.1
            else:
                move_duration = 60  # デフォルト60秒
            
            for j in range(move_frames):
                # 移動中の時間を計算
                move_progress = (j / move_frames) * move_duration
                current_timestamp = df_filtered['timestamp'].iloc[i] + pd.Timedelta(seconds=stay_duration_seconds + move_progress)
                
                frames_data.append({
                    'x': x_path[j],
                    'y': y_path[j],
                    'timestamp': current_timestamp,
                    'location': f"{df_filtered['location'].iloc[i]} → {df_filtered['location'].iloc[i+1]}",
                    'trail_x': df_filtered['x_noisy'].iloc[:i+1].tolist() + x_path[:j+1].tolist(),
                    'trail_y': df_filtered['y_noisy'].iloc[:i+1].tolist() + y_path[:j+1].tolist()
                })
        else:
            # 最後の位置
            frames_data.append({
                'x': df_filtered['x_noisy'].iloc[i],
                'y': df_filtered['y_noisy'].iloc[i],
                'timestamp': df_filtered['timestamp'].iloc[i],
                'location': df_filtered['location'].iloc[i],
                'trail_x': df_filtered['x_noisy'].tolist(),
                'trail_y': df_filtered['y_noisy'].tolist()
            })
    
    # アニメーション更新関数
    def update(frame):
        if frame >= len(frames_data):
            return []
        
        data = frames_data[frame]
        
        # 軌跡を更新
        line.set_data(data['trail_x'], data['trail_y'])
        
        # 猫の位置を更新
        if cat_img is not None and cat_annotation is not None:
            cat_annotation.xybox = (data['x'], data['y'])
        elif cat_point is not None:
            cat_point.set_data([data['x']], [data['y']])
        
        # テキストを更新
        time_text.set_text(f"時刻: {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        location_text.set_text(f"場所: {data['location']}")
        
        if cat_img is not None and cat_annotation is not None:
            return [line, cat_annotation, time_text, location_text]
        elif cat_point is not None:
            return [line, cat_point, time_text, location_text]
        else:
            return [line, time_text, location_text]
    
    # アニメーション作成
    # インターバルも速度係数で調整（speed_factor > 1 で短く、< 1 で長く）
    interval = int(DEFAULT_INTERVAL / speed_factor)
    anim = animation.FuncAnimation(fig, update, frames=len(frames_data), 
                                 interval=interval, blit=True, repeat=True)
    
    # アニメーションを保存する場合
    if output_format and output_file:
        print(f"アニメーションを {output_file} として保存中...")
        # FPSの決定（指定されていない場合はデフォルト値を速度係数で調整）
        save_fps = fps if fps is not None else int(DEFAULT_FPS * speed_factor)
        
        if output_format == 'gif':
            try:
                anim.save(output_file, writer='pillow', fps=save_fps)
                print(f"GIFファイルを保存しました: {output_file} (FPS: {save_fps})")
            except Exception as e:
                print(f"GIF保存エラー: {e}")
                print("pillowがインストールされていない可能性があります。'pip install pillow' を実行してください。")
        elif output_format == 'mp4':
            try:
                anim.save(output_file, writer='ffmpeg', fps=save_fps, bitrate=1800)
                print(f"MP4ファイルを保存しました: {output_file} (FPS: {save_fps})")
            except Exception as e:
                print(f"MP4保存エラー: {e}")
                print("ffmpegがインストールされていない可能性があります。")
                print("macOS: 'brew install ffmpeg'")
                print("Ubuntu/Debian: 'sudo apt-get install ffmpeg'")
                print("Windows: ffmpegをダウンロードしてPATHに追加してください。")
    
    plt.tight_layout()
    
    # 保存のみの場合は表示しない
    if not (output_format and output_file):
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Animate cat walk from log file.')
    parser.add_argument('log_file', nargs='?', default='rssi_inference_log.csv',
                        help='Path to the log file (default: rssi_inference_log.csv)')
    parser.add_argument('--save-gif', action='store_true',
                        help='Save animation as GIF file')
    parser.add_argument('--save-mp4', action='store_true',
                        help='Save animation as MP4 file')
    parser.add_argument('--output', '-o', type=str,
                        help='Output filename (default: cat_walk_animation.gif/mp4)')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                        help='Animation speed factor (default: 1.0, >1 for faster, <1 for slower)')
    parser.add_argument('--fps', type=int,
                        help='Frames per second for saved video (default: auto-calculated based on speed)')
    args = parser.parse_args()

    # 出力形式とファイル名を決定
    output_format = None
    output_file = None
    
    if args.save_gif:
        output_format = 'gif'
        output_file = args.output if args.output else 'cat_walk_animation.gif'
    elif args.save_mp4:
        output_format = 'mp4'
        output_file = args.output if args.output else 'cat_walk_animation.mp4'
    
    main(args.log_file, output_format, output_file, args.speed, args.fps)
