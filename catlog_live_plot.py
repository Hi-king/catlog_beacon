#!/usr/bin/env python3
"""
cat_device_live_plot.py   (macOS 用)

 - 名前が "Cat" で始まる BLE デバイスを自動認識
 - 接続は行わず、広告パケットの RSSI を取り続ける
 - 計測開始からの全履歴をリアルタイム描画
"""

import asyncio, threading, time, sys, csv, atexit
from collections import deque
from datetime import datetime

from bleak import BleakScanner
import matplotlib.pyplot as plt

# ──────────────── ここだけ変更しても良いパラメータ ────────────────
TARGET_PREFIX   = "Cat"      # デバイス名のプレフィクス
SCAN_INTERVAL   = 0.03        # スキャナループのウェイト [s]
PLOT_INTERVAL   = 0.25       # グラフ更新間隔 [s]
Y_LIM           = (-100, -30)  # RSSI 表示範囲
CSV_FILENAME    = "rssi_log.csv" # 出力CSVファイル名
# ────────────────────────────────────────────────

# ── 共有バッファ（スレッド間） ───────────────────────────
t0         = time.time()

# --- CSV ファイル準備 ---
try:
    csv_file = open(CSV_FILENAME, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # ヘッダーに datetime を追加
    csv_writer.writerow(["datetime", "time_s", "rssi_dbm"])
    print(f"[INFO] Writing RSSI data to '{CSV_FILENAME}'")

    @atexit.register
    def close_csv():
        if csv_file and not csv_file.closed:
            csv_file.close()
            print(f"[INFO] CSV file '{CSV_FILENAME}' closed.")

except IOError as e:
    print(f"[ERROR] Could not open or write to CSV file '{CSV_FILENAME}': {e}")
    sys.exit(1)

time_buf   = []            # 全履歴を溜める（長時間なら deque(maxlen=…) 推奨）
rssi_buf   = []
buf_lock   = threading.Lock()

# ── BLE スキャン：別スレッドで回す ───────────────────────
def start_ble_thread():
    asyncio.run(ble_loop())

async def ble_loop():
    target_addr = None
    found_event = asyncio.Event()

    def detection_cb(device, adv):
        nonlocal target_addr
        # ① デバイス特定
        if target_addr is None:
            if device.name and device.name.startswith(TARGET_PREFIX):
                target_addr = device.address
                print(f"[INFO] TARGET FOUND  →  {device.address} ({device.name})")
                found_event.set()

        # ② RSSI 取得 & CSV 書き込み
        if device.address == target_addr:
            # 絶対時刻と相対時刻を取得
            now = datetime.now()
            absolute_time_str = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # ミリ秒まで
            relative_time = time.time() - t0
            rssi = device.rssi

            with buf_lock:
                time_buf.append(relative_time) # プロット用は相対時間のまま
                rssi_buf.append(rssi)
                # CSV に追記 (絶対時刻、相対時間、RSSI)
                row_data = [absolute_time_str, f"{relative_time:.3f}", rssi]
                try:
                    csv_writer.writerow(row_data)
                    # 標準出力にも表示
                    print(f"Logged: {row_data[0]}, Time: {row_data[1]}s, RSSI: {row_data[2]} dBm")
                except Exception as e:
                    # 書き込みエラーが発生してもプログラムは止めない（ログだけ出す）
                    print(f"[WARN] Failed to write to CSV: {e}")

    scanner = BleakScanner(detection_cb)

    # --- スキャン開始 ---
    await scanner.start()
    # 見つかるまで（見つかったらすぐ戻る）
    await asyncio.wait_for(found_event.wait(), timeout=None)

    # 見つかった後はずっとスキャン継続
    try:
        while True:
            await asyncio.sleep(SCAN_INTERVAL)
    finally:
        await scanner.stop()

# スキャナスレッド起動
threading.Thread(target=start_ble_thread, daemon=True).start()

# ── Matplotlib リアルタイムプロット ──────────────────────
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [],  marker='o', markersize=3, lw=1.5, label="Target RSSI")
ax.set_xlabel("Elapsed time [s]")
ax.set_ylabel("RSSI [dBm]")
ax.set_ylim(*Y_LIM)
ax.set_xlim(0, 10)
ax.set_title(f'BLE RSSI – devices starting with "{TARGET_PREFIX}"')
ax.legend(loc="upper left")

while True:
    with buf_lock:
        if time_buf:
            line.set_data(time_buf, rssi_buf)
            ax.set_xlim(0, max(10, time_buf[-1] + 1))
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    time.sleep(PLOT_INTERVAL)
