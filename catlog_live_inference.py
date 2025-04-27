#!/usr/bin/env python3
"""
catlog_live_inference.py (macOS 用)

 - 名前が "Cat" で始まる BLE デバイスを自動認識し、RSSIを記録・プロット。
 - 1分ごとに過去1分間のデータから場所を推論し、条件を満たし場所が変われば結果をSlackに投稿。
"""

import asyncio
import threading
import time
import sys
import csv
import atexit
import argparse
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from bleak import BleakScanner
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語表示用

# ──────────────── 設定 ────────────────
TARGET_PREFIX   = "Cat"      # デバイス名のプレフィクス
SCAN_INTERVAL   = 0.03       # スキャナループのウェイト [s]
PLOT_INTERVAL   = 0.25       # グラフ更新間隔 [s]
INFERENCE_INTERVAL_SEC = 60   # 推論実行間隔 [s]
CONSECUTIVE_PREDICTIONS_FOR_SLACK = 3 # Slack通知に必要な連続同一予測回数
SLACK_CONFIDENCE_THRESHOLD = 0.60 # Slack通知に必要な信頼度(確率)の閾値 (0.0 ~ 1.0)
Y_LIM           = (-100, -30) # RSSI 表示範囲
CSV_FILENAME    = "rssi_inference_log.csv" # 出力CSVファイル名
MODEL_BUNDLE_PATH = "location_classification_bundle.joblib" # 学習済みモデルバンドル
PLOT_WINDOW_SEC = 600 # プロット表示期間（秒）

# ──────────────── グローバル変数 ────────────────
t0 = time.time()
buf_lock = threading.Lock()
# プロット&CSV用バッファ
time_buf = deque(maxlen=int(PLOT_WINDOW_SEC / SCAN_INTERVAL) if SCAN_INTERVAL > 0 else 10000)
rssi_buf = deque(maxlen=int(PLOT_WINDOW_SEC / SCAN_INTERVAL) if SCAN_INTERVAL > 0 else 10000)
# 推論用データバッファ
inference_data_window = deque(maxlen=int((INFERENCE_INTERVAL_SEC + 10) / SCAN_INTERVAL) if SCAN_INTERVAL > 0 else 500) # 余裕を持たせる
last_inference_time = time.time()
last_predicted_location = "不明" # プロット表示用
prediction_history = deque(maxlen=CONSECUTIVE_PREDICTIONS_FOR_SLACK) # Slack通知判定用の予測履歴
# Slack通知状態管理
last_notified_location = None # 最後に通知した場所
last_notified_time = None     # 最後に通知した時刻 (time.time())

# --- コマンドライン引数 ---
parser = argparse.ArgumentParser(description="リアルタイムRSSIプロット＆場所推論＆Slack通知")
parser.add_argument("--slack-token", required=True, help="Slack Bot Token")
parser.add_argument("--slack-channel", required=True, help="Slack Channel Name (e.g., #general) or ID")
args = parser.parse_args()

# --- モデルバンドルの読み込み ---
try:
    bundle_path = Path(MODEL_BUNDLE_PATH)
    if not bundle_path.exists():
        print(f"[ERROR] Model bundle not found: {MODEL_BUNDLE_PATH}")
        sys.exit(1)
    bundle = joblib.load(bundle_path)
    model = bundle['model']
    scaler = bundle['scaler']
    class_names = bundle['classes']
    feature_names = bundle['features']
    min_points_per_window = bundle['min_points_per_window']
    print(f"[INFO] Model bundle loaded successfully from {MODEL_BUNDLE_PATH}")
    print(f"  - Classes: {class_names}")
    print(f"  - Features: {feature_names}")
    print(f"  - Min points for inference: {min_points_per_window}")
except Exception as e:
    print(f"[ERROR] Failed to load model bundle: {e}")
    sys.exit(1)

# --- Slackクライアント初期化 ---
slack_client = WebClient(token=args.slack_token)
try:
    response = slack_client.auth_test()
    print(f"[INFO] Slack client initialized successfully for user {response['user']}")
except SlackApiError as e:
    print(f"[ERROR] Failed to initialize Slack client: {e.response['error']}")
    sys.exit(1)

# --- CSV ファイル準備 ---
try:
    csv_file = open(CSV_FILENAME, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["datetime", "time_s", "rssi_dbm", "predicted_location"]) # ヘッダー
    print(f"[INFO] Writing log data to '{CSV_FILENAME}'")

    @atexit.register
    def close_csv():
        if csv_file and not csv_file.closed:
            csv_file.close()
            print(f"[INFO] CSV file '{CSV_FILENAME}' closed.")

except IOError as e:
    print(f"[ERROR] Could not open or write to CSV file '{CSV_FILENAME}': {e}")
    sys.exit(1)

# ───────────────── 関数 ─────────────────

def post_to_slack(channel, text):
    """Slackにメッセージを投稿する"""
    try:
        response = slack_client.chat_postMessage(channel=channel, text=text)
        # 成功ログは呼び出し元で出力
    except SlackApiError as e:
        print(f"[ERROR] Failed to post to Slack: {e.response['error']}")

def format_duration(seconds):
    """秒数を「X分Y秒」の形式にフォーマットする"""
    if seconds < 0:
        return "不明な時間"
    minutes, sec = divmod(int(seconds), 60)
    if minutes > 0:
        return f"約{minutes}分{sec}秒"
    else:
        return f"約{sec}秒"

def extract_inference_features(data_list):
    """推論用の特徴量を抽出する"""
    if not data_list or len(data_list) < min_points_per_window:
        return None

    rssi_values = [item[1] for item in data_list]
    rssi_series = pd.Series(rssi_values)

    features = {
        'rssi_mean': rssi_series.mean(),
        'rssi_std': rssi_series.std(),
        'rssi_min': rssi_series.min(),
        'rssi_max': rssi_series.max(),
        'rssi_median': rssi_series.median(),
        'rssi_range': rssi_series.max() - rssi_series.min(),
    }
    feature_df = pd.DataFrame([features])[feature_names]
    feature_df['rssi_std'].fillna(0, inplace=True)
    return feature_df

def run_inference():
    """データウィンドウから特徴量を抽出し、推論を実行、条件を満たせばSlackに投稿"""
    global last_predicted_location, prediction_history, last_notified_location, last_notified_time
    current_run_time = time.time() # この推論実行開始時刻
    one_minute_ago = current_run_time - INFERENCE_INTERVAL_SEC

    recent_data = [item for item in inference_data_window if item[0] >= one_minute_ago]

    if len(recent_data) >= min_points_per_window:
        print(f"[INFO] Running inference with {len(recent_data)} data points from the last {INFERENCE_INTERVAL_SEC} seconds...")
        feature_df = extract_inference_features(recent_data)

        if feature_df is not None:
            try:
                features_scaled = scaler.transform(feature_df)
                probabilities = model.predict_proba(features_scaled)[0]
                predicted_index = np.argmax(probabilities)
                predicted_location = class_names[predicted_index]
                predicted_probability = probabilities[predicted_index]

                print(f"[INFO] Predicted location: {predicted_location} (Prob: {predicted_probability:.2f})")
                last_predicted_location = predicted_location # プロット用は常に更新

                # --- Slack通知判定ロジック ---
                prediction_history.append(predicted_location)

                if len(prediction_history) == CONSECUTIVE_PREDICTIONS_FOR_SLACK:
                    first_prediction = prediction_history[0]
                    is_consecutive = all(p == first_prediction for p in prediction_history)

                    if is_consecutive:
                        confirmed_location = first_prediction
                        confirmed_probability = predicted_probability # 最新の確率を使用
                        confirmed_time = current_run_time # 確定した時刻

                        # 信頼度が閾値以上か？
                        if confirmed_probability >= SLACK_CONFIDENCE_THRESHOLD:
                            print(f"[INFO] Location '{confirmed_location}' confirmed (Confidence: {confirmed_probability:.2f})")

                            # 最後に通知した場所から変わったか？ (初回通知も含む)
                            if confirmed_location != last_notified_location:
                                duration_str = ""
                                if last_notified_location is not None and last_notified_time is not None:
                                    duration_seconds = confirmed_time - last_notified_time
                                    duration_str = f"（「{last_notified_location}」に{format_duration(duration_seconds)}滞在）"

                                slack_message = f"猫様は現在「{confirmed_location}」に移動しました {duration_str}(信頼度: {confirmed_probability:.0%})"
                                post_to_slack(args.slack_channel, slack_message)
                                print(f"[INFO] Slack notification condition met (location changed). Posted: {slack_message}")

                                # 通知状態を更新
                                last_notified_location = confirmed_location
                                last_notified_time = confirmed_time
                            else:
                                # 場所は確定したが、前回通知から変わっていない
                                print(f"[INFO] Location '{confirmed_location}' confirmed, but no change since last notification. No Slack post.")

                            # 場所が確定したら（通知有無に関わらず）履歴をクリア
                            prediction_history.clear()
                            print("[INFO] Prediction history cleared after confirmation.")

                        else:
                            # 連続だが信頼度が低い
                            print(f"[INFO] Consecutive prediction '{confirmed_location}', but confidence ({confirmed_probability:.2f}) is below threshold ({SLACK_CONFIDENCE_THRESHOLD}). Not confirmed yet.")
                            # 履歴はクリアしない（次の推論で閾値を超える可能性を待つ）
                    else:
                        # 履歴は溜まったが連続ではない
                        print(f"[INFO] Prediction history is full but not consecutive: {list(prediction_history)}. No confirmation.")
                        # 履歴はdequeにより自動で古いものが削除される
                else:
                    # 履歴がまだ溜まっていない
                     print(f"[INFO] Prediction history count ({len(prediction_history)}/{CONSECUTIVE_PREDICTIONS_FOR_SLACK}) not met yet.")
                # --- Slack通知判定ここまで ---

            except Exception as e:
                print(f"[ERROR] Error during inference or Slack posting check: {e}")
        else:
             print("[INFO] Feature extraction failed (likely not enough data).")
    else:
        print(f"[INFO] Skipping inference: Only {len(recent_data)} points in the last {INFERENCE_INTERVAL_SEC} seconds (min: {min_points_per_window}).")


# ── BLE スキャン：別スレッドで回す ───────────────────────
def start_ble_thread():
    asyncio.run(ble_loop())

async def ble_loop():
    global last_inference_time
    target_addr = None
    found_event = asyncio.Event()

    def detection_cb(device, adv):
        nonlocal target_addr
        global last_inference_time, last_predicted_location

        if target_addr is None:
            if device.name and device.name.startswith(TARGET_PREFIX):
                target_addr = device.address
                print(f"[INFO] TARGET FOUND  →  {device.address} ({device.name})")
                found_event.set()

        if device.address == target_addr:
            now = datetime.now()
            current_timestamp = now.timestamp()
            relative_time = current_timestamp - t0
            rssi = device.rssi
            absolute_time_str = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            with buf_lock:
                time_buf.append(relative_time)
                rssi_buf.append(rssi)
                inference_data_window.append((current_timestamp, rssi))

                row_data = [absolute_time_str, f"{relative_time:.3f}", rssi, last_predicted_location] # CSVには最新の推論を記録
                try:
                    csv_writer.writerow(row_data)
                except Exception as e:
                    print(f"[WARN] Failed to write to CSV: {e}")

            current_time_check = time.time()
            if current_time_check - last_inference_time >= INFERENCE_INTERVAL_SEC:
                run_inference()
                last_inference_time = current_time_check # 次のインターバル開始時刻を更新

    scanner = BleakScanner(detection_cb)

    await scanner.start()
    print("[INFO] Scanning for devices...")
    try:
        await asyncio.wait_for(found_event.wait(), timeout=None)
        print("[INFO] Target found. Continuously scanning and inferring...")
        while True:
            await asyncio.sleep(SCAN_INTERVAL)
    except asyncio.CancelledError:
        print("[INFO] Scan task cancelled.")
    finally:
        await scanner.stop()
        print("[INFO] Scanner stopped.")

ble_thread = threading.Thread(target=start_ble_thread, daemon=True)
ble_thread.start()

# ── Matplotlib リアルタイムプロット ──────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot([], [], marker='o', markersize=2, lw=1, label="Target RSSI")
ax.set_xlabel("Elapsed time [s]")
ax.set_ylabel("RSSI [dBm]")
ax.set_ylim(*Y_LIM)
ax.legend(loc="upper left")
# タイトルには最後に「通知された」場所を表示する方が混乱が少ないかも？
# title_text = ax.set_title(f'BLE RSSI – "{TARGET_PREFIX}" / Last notified: {last_notified_location or "N/A"}')
title_text = ax.set_title(f'BLE RSSI – "{TARGET_PREFIX}" / Predicted: {last_predicted_location}') # とりあえず最新推論のまま


print("[INFO] Plotting started. Close the plot window to exit.")

try:
    while True:
        with buf_lock:
            if time_buf:
                current_max_time = time_buf[-1]
                plot_start_time = max(0, current_max_time - PLOT_WINDOW_SEC)
                ax.set_xlim(plot_start_time, current_max_time + PLOT_WINDOW_SEC * 0.05)

                line.set_data(list(time_buf), list(rssi_buf))
                # タイトル更新 (最新推論のまま)
                title_text.set_text(f'BLE RSSI – "{TARGET_PREFIX}" / Predicted: {last_predicted_location}')
                # # タイトル更新 (最後に通知された場所を表示する場合)
                # current_notified_str = last_notified_location if last_notified_location is not None else "N/A"
                # title_text.set_text(f'BLE RSSI – "{TARGET_PREFIX}" / Last notified: {current_notified_str}')


        fig.canvas.draw_idle()
        if not plt.fignum_exists(fig.number):
            print("[INFO] Plot window closed. Exiting...")
            break
        fig.canvas.flush_events()
        time.sleep(PLOT_INTERVAL)

except KeyboardInterrupt:
    print("[INFO] KeyboardInterrupt received. Exiting...")
finally:
    print("[INFO] Cleanup finished.")
