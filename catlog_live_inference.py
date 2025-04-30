#!/usr/bin/env python3
"""
catlog_live_inference.py (macOS 用)

 - 名前が "Cat" で始まる BLE デバイスを自動認識し、RSSIを記録・プロット。
 - 1分ごとに過去1分間のデータから場所を推論し、条件を満たし場所が変われば結果をSlackに投稿。
 - 場所変更通知時、スレッドに直近30分の推論履歴を投稿。
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
from typing import Deque, List, Tuple, Optional, Any # ★ 型ヒント用

import pandas as pd
import numpy as np
# import joblib # model_utils で使用
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from bleak import BleakScanner
import matplotlib.pyplot as plt
import japanize_matplotlib # type: ignore # 日本語表示用

# --- ライブラリからのインポート ---
from blelocation.model_utils import load_model_bundle
from blelocation.feature_engineering import extract_inference_features

# ──────────────── 設定 ────────────────
TARGET_PREFIX   = "Cat"      # デバイス名のプレフィクス
SCAN_INTERVAL   = 0.03       # スキャナループのウェイト [s]
PLOT_INTERVAL   = 0.25       # グラフ更新間隔 [s]
INFERENCE_INTERVAL_SEC = 60   # 推論実行間隔 [s]
CONSECUTIVE_PREDICTIONS_FOR_SLACK = 3 # Slack通知に必要な連続同一予測回数
SLACK_CONFIDENCE_THRESHOLD = 0.50 # Slack通知に必要な信頼度(確率)の閾値 (0.0 ~ 1.0)
INFERENCE_LOG_HISTORY_MINUTES = 30 # Slackスレッド投稿用の履歴保持期間(分)
Y_LIM           = (-100, -30) # RSSI 表示範囲
CSV_FILENAME    = "rssi_inference_log.csv" # 出力CSVファイル名
MODEL_BUNDLE_PATH = "location_classification_bundle.joblib" # 学習済みモデルバンドル
PLOT_WINDOW_SEC = 600 # プロット表示期間（秒）

# ──────────────── グローバル変数 ────────────────
t0 = time.time()
buf_lock = threading.Lock()
# プロット&CSV用バッファ
time_buf: Deque[float] = deque(maxlen=int(PLOT_WINDOW_SEC / SCAN_INTERVAL) if SCAN_INTERVAL > 0 else 10000)
rssi_buf: Deque[int] = deque(maxlen=int(PLOT_WINDOW_SEC / SCAN_INTERVAL) if SCAN_INTERVAL > 0 else 10000)
# 推論用データバッファ (タイムスタンプ, RSSI)
inference_data_window: Deque[Tuple[float, int]] = deque(maxlen=int((INFERENCE_INTERVAL_SEC + 10) / SCAN_INTERVAL) if SCAN_INTERVAL > 0 else 500) # 余裕を持たせる
# 推論ログ履歴バッファ (Slackスレッド投稿用)
inference_log_history_maxlen = int(INFERENCE_LOG_HISTORY_MINUTES * 60 / INFERENCE_INTERVAL_SEC) if INFERENCE_INTERVAL_SEC > 0 else 30
inference_log_history: Deque[Tuple[datetime, str, float]] = deque(maxlen=inference_log_history_maxlen) # (datetime, location, probability)
last_inference_time = time.time()
last_predicted_location: str = "不明" # プロット表示用
prediction_history: Deque[str] = deque(maxlen=CONSECUTIVE_PREDICTIONS_FOR_SLACK) # Slack通知判定用の予測履歴
# Slack通知状態管理
last_notified_location = None # 最後に通知した場所
last_notified_time = None     # 最後に通知した時刻 (time.time())

# --- コマンドライン引数 ---
parser = argparse.ArgumentParser(description="リアルタイムRSSIプロット＆場所推論（Slack通知はオプション）")
parser.add_argument("--slack-token", required=False, help="Slack Bot Token (通知する場合に必須)")
parser.add_argument("--slack-channel", required=False, help="Slack Channel Name (e.g., #general) or ID (通知する場合に必須)")
args = parser.parse_args()

# --- Slack 利用判定 ---
USE_SLACK = args.slack_token is not None and args.slack_channel is not None
if USE_SLACK:
    print("[INFO] Slack notification enabled.")
else:
    print("[INFO] Slack notification disabled.")


# --- モデルバンドルの読み込み ---
bundle = load_model_bundle(MODEL_BUNDLE_PATH)
if bundle is None:
    sys.exit(1) # load_model_bundle 内でエラーメッセージ表示済み

model = bundle['model']
scaler = bundle['scaler']
class_names = bundle['classes']
feature_names = bundle['features']
min_points_per_window = bundle['min_points_per_window']
print(f"  - Classes: {class_names}")
print(f"  - Features: {feature_names}")
print(f"  - Min points for inference: {min_points_per_window}")

# --- Slackクライアント初期化 (必要な場合のみ) ---
slack_client = None
if USE_SLACK:
    slack_client = WebClient(token=args.slack_token)
    try:
        response = slack_client.auth_test()
        print(f"[INFO] Slack client initialized successfully for user {response['user']}")
    except SlackApiError as e:
        print(f"[ERROR] Failed to initialize Slack client: {e.response['error']}")
        print("[INFO] Proceeding without Slack notifications.")
        USE_SLACK = False # 初期化失敗したら Slack は使わない
        slack_client = None # クライアントも None に戻す

# --- CSV ファイル準備 ---
try:
    # ファイルが存在するか確認し、ヘッダーが必要か判断
    csv_path = Path(CSV_FILENAME)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    # 追記モード ('a') でファイルを開く
    csv_file = open(CSV_FILENAME, 'a', newline='')
    csv_writer = csv.writer(csv_file)

    if write_header:
        csv_writer.writerow(["datetime", "time_s", "rssi_dbm", "predicted_location"]) # ヘッダー
        print(f"[INFO] Writing header and starting log to '{CSV_FILENAME}'")
    else:
        print(f"[INFO] Appending log data to existing '{CSV_FILENAME}'")


    @atexit.register
    def close_csv():
        if csv_file and not csv_file.closed:
            csv_file.close()
            print(f"[INFO] CSV file '{CSV_FILENAME}' closed.")

except IOError as e:
    print(f"[ERROR] Could not open or write to CSV file '{CSV_FILENAME}': {e}")
    sys.exit(1)

# ───────────────── 関数 ─────────────────

def post_to_slack(channel, text, thread_ts=None):
    """Slackにメッセージを投稿する。成功したらレスポンスを、失敗したらNoneを返す"""
    if not USE_SLACK or slack_client is None:
        print("[DEBUG] Slack is disabled or client not initialized. Skipping post.")
        return None # Slackが無効なら何もしない

    try:
        response = slack_client.chat_postMessage(
            channel=channel,
            text=text,
            thread_ts=thread_ts # スレッド投稿用に追加
        )
        return response # 成功時はレスポンスを返す
    except SlackApiError as e:
        print(f"[ERROR] Failed to post to Slack: {e.response['error']}")
        return None # 失敗時はNoneを返す

def format_duration(seconds):
    """秒数を「X分Y秒」の形式にフォーマットする"""
    if seconds is None or seconds < 0:
        return "不明な時間"
    minutes, sec = divmod(int(seconds), 60)
    if minutes > 0:
        return f"約{minutes}分{sec}秒"
    else:
        return f"約{sec}秒"


def run_inference():
    """データウィンドウから特徴量を抽出し、推論を実行、条件を満たせばSlackに投稿＆履歴投稿"""
    global last_predicted_location, prediction_history, inference_log_history
    global last_notified_location, last_notified_time
    current_run_time = time.time() # この推論実行開始時刻
    now_dt = datetime.now() # 履歴記録用
    one_minute_ago = current_run_time - INFERENCE_INTERVAL_SEC

    recent_data = [item for item in inference_data_window if item[0] >= one_minute_ago]

    if len(recent_data) >= min_points_per_window:
        print(f"[INFO] Running inference with {len(recent_data)} data points from the last {INFERENCE_INTERVAL_SEC} seconds...")
        # 入力形式をDataFrameに変換
        recent_data_df = pd.DataFrame(recent_data, columns=['timestamp', 'rssi_dbm'])
        # ライブラリ関数を呼び出す
        feature_df = extract_inference_features(
            data=recent_data_df,
            rssi_col='rssi_dbm', # RSSIデータが含まれる列名
            feature_names=feature_names, # モデルが学習した特徴量名を指定
            min_points_required=min_points_per_window # 最小ポイント数を指定
        )

        if feature_df is not None:
            try:
                features_scaled = scaler.transform(feature_df)
                probabilities = model.predict_proba(features_scaled)[0]
                predicted_index = np.argmax(probabilities)
                predicted_location = class_names[predicted_index]
                predicted_probability = probabilities[predicted_index]

                print(f"[INFO] Predicted location: {predicted_location} (Prob: {predicted_probability:.2f})")
                last_predicted_location = predicted_location # プロット用は常に更新

                # 推論ログ履歴に追加
                inference_log_history.append((now_dt, predicted_location, predicted_probability))

                # --- Slack通知判定ロジック ---
                # 信頼度が閾値以上の場合のみ連続予測履歴に追加
                if predicted_probability >= SLACK_CONFIDENCE_THRESHOLD:
                    prediction_history.append(predicted_location)
                    print(f"[DEBUG] Added '{predicted_location}' to prediction_history (Prob: {predicted_probability:.2f} >= {SLACK_CONFIDENCE_THRESHOLD})")
                else:
                    # 信頼度が低い場合は履歴に追加せず、連続性をチェックしない
                    # (連続履歴が途切れることになる)
                    print(f"[DEBUG] Skipped adding '{predicted_location}' to prediction_history due to low confidence (Prob: {predicted_probability:.2f} < {SLACK_CONFIDENCE_THRESHOLD})")

                # 連続予測回数に達した場合のみ確認処理へ進む
                if len(prediction_history) == CONSECUTIVE_PREDICTIONS_FOR_SLACK:
                    first_prediction = prediction_history[0]
                    is_consecutive = all(p == first_prediction for p in prediction_history)

                    if is_consecutive:
                        confirmed_location = first_prediction
                        # 連続が確認された時点での最新の確率を使用 (履歴には古い確率も含まれる可能性があるため)
                        # confirmed_probability = predicted_probability # この行は不要 (閾値チェックで既に実施済み)
                        confirmed_time = current_run_time # 確定した時刻

                        # 信頼度が閾値以上か？ (履歴追加時にチェック済みだが念のため)
                        # ※ 信頼度が低い場合は prediction_history に追加されないため、この if 文は常に True になるはず
                        if predicted_probability >= SLACK_CONFIDENCE_THRESHOLD:
                            print(f"[INFO] Location '{confirmed_location}' confirmed (Confidence: {predicted_probability:.2f})")

                            # 最後に通知した場所から変わったか？ (初回通知も含む)
                            if confirmed_location != last_notified_location:
                                duration_seconds = None
                                if last_notified_location is not None and last_notified_time is not None:
                                    duration_seconds = confirmed_time - last_notified_time
                                duration_str = f"（「{last_notified_location}」に{format_duration(duration_seconds)}滞在）" if duration_seconds is not None else ""
                                slack_message = f"猫様は現在「{confirmed_location}」に移動しました {duration_str}(信頼度: {predicted_probability:.0%})"

                                # --- Slack通知実行 (有効な場合のみ) ---
                                main_post_response = None
                                message_ts = None
                                if USE_SLACK:
                                    print("[INFO] Posting location change to Slack...")
                                    main_post_response = post_to_slack(args.slack_channel, slack_message)
                                    if main_post_response and main_post_response.get("ok"):
                                        print(f"[INFO] Slack notification posted (location changed): {slack_message}")
                                        message_ts = main_post_response.get("ts")
                                    else:
                                        print("[WARN] Failed to post main Slack notification.")
                                else:
                                    print(f"[INFO] Slack disabled. Skipping notification for: {slack_message}")

                                # --- Slack通知が成功した場合、またはSlackが無効な場合に状態を更新 ---
                                # (Slackが無効でも場所が変わったという事実は記録するため)
                                if (USE_SLACK and main_post_response and main_post_response.get("ok")) or not USE_SLACK:

                                    # --- スレッドへの履歴投稿 (Slack通知成功時のみ) ---
                                    if USE_SLACK and message_ts:
                                        thread_history_limit = now_dt - timedelta(minutes=INFERENCE_LOG_HISTORY_MINUTES)
                                        history_lines = []
                                        for dt, loc, prob in reversed(inference_log_history): # 新しい順に
                                            if dt >= thread_history_limit:
                                                history_lines.append(f"- {dt.strftime('%H:%M:%S')}: {loc} ({prob:.0%})")
                                            else:
                                                break # 期間外になったら終了

                                        if history_lines:
                                            thread_message = f"直近{INFERENCE_LOG_HISTORY_MINUTES}分間の推論履歴:\n" + "\n".join(reversed(history_lines)) # 古い順に戻す
                                            thread_post_response = post_to_slack(args.slack_channel, thread_message, thread_ts=message_ts)
                                            if thread_post_response and thread_post_response.get("ok"):
                                                print(f"[INFO] Posted inference history to the thread (ts: {message_ts})")
                                            else:
                                                print(f"[WARN] Failed to post history to the thread (ts: {message_ts})")
                                        else:
                                            print("[INFO] No recent inference history to post to the thread.")
                                    elif USE_SLACK and not message_ts:
                                        print("[WARN] Could not get message timestamp ('ts') to post history to thread.")
                                    # --- スレッド投稿ここまで ---

                                    # 状態を更新 (場所が変わった事実は記録)
                                    last_notified_location = confirmed_location
                                    last_notified_time = confirmed_time
                                    # ★★★ 場所変更が確認された後 (Slack通知有無に関わらず) に履歴をクリア ★★★
                                    prediction_history.clear()
                                    print("[INFO] Prediction history cleared after location change was confirmed.")
                                # else: # Slack有効でメイン投稿失敗した場合 -> 状態は更新せず、履歴もクリアしない

                            else:
                                # 場所は確定したが、前回通知から変わっていない (Slack通知不要)
                                print(f"[INFO] Location '{confirmed_location}' confirmed, but no change since last notification. No Slack post.")
                                # ★ 場所が変わっていない場合は履歴をクリアしない ★

                        # else: # ★ この else ブロックは不要 (閾値チェックは履歴追加時に実施済み)
                            # 連続だが信頼度が低い
                            # print(f"[INFO] Consecutive prediction '{confirmed_location}', but confidence ({predicted_probability:.2f}) is below threshold ({SLACK_CONFIDENCE_THRESHOLD}). Not confirmed yet.")
                    else:
                        # 履歴は溜まったが連続ではない (閾値チェックにより通常ここには来ないはず)
                        print(f"[INFO] Prediction history is full but not consecutive: {list(prediction_history)}. No confirmation.")
                # 連続回数に達していない場合のログは削除 (不要なため)
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
            rssi = adv.rssi # device.rssi から変更
            absolute_time_str = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            with buf_lock:
                time_buf.append(relative_time)
                rssi_buf.append(rssi)
                inference_data_window.append((current_timestamp, rssi))

                row_data = [absolute_time_str, f"{relative_time:.3f}", rssi, last_predicted_location] # CSVには最新の推論を記録
                try:
                    csv_writer.writerow(row_data)
                    csv_file.flush() # バッファを即座にディスクに書き込む
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
title_text = ax.set_title(f'BLE RSSI – "{TARGET_PREFIX}" / Predicted: {last_predicted_location}')


print("[INFO] Plotting started. Close the plot window to exit.")

try:
    while True:
        with buf_lock:
            if time_buf:
                current_max_time = time_buf[-1]
                plot_start_time = max(0, current_max_time - PLOT_WINDOW_SEC)
                ax.set_xlim(plot_start_time, current_max_time + PLOT_WINDOW_SEC * 0.05)

                line.set_data(list(time_buf), list(rssi_buf))
                title_text.set_text(f'BLE RSSI – "{TARGET_PREFIX}" / Predicted: {last_predicted_location}')


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
