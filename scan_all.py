# scan_detail.py
import asyncio
import csv
import datetime
from collections import defaultdict
from typing import List, Dict, Any
from bleak import BleakScanner

def pretty_mfg(mfg_dict: Dict[int, bytes]) -> str:
    return ",".join(f"{cid:04X}:{data.hex()}" for cid, data in mfg_dict.items()) or "-"

devices_data: List[Dict[str, Any]] = []

def cb(dev, adv):
    devices_data.append(
        {
            "address": dev.address,
            "rssi": dev.rssi,
            "name": dev.name,
            "manufacturer_data": adv.manufacturer_data,
            "service_uuids": adv.service_uuids,
        }
    )

async def main():
    scanner = BleakScanner(cb)
    print("Scanning for 60 seconds...")
    await scanner.start()
    try:
        await asyncio.sleep(60)
    finally:
        await scanner.stop()
        print("Scan finished.")

    # デバイスアドレスごとにRSSIとmanufacturer_dataをグループ化
    data_by_device: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"rssi_list": [], "manufacturer_data": {}})
    for dev_info in devices_data:
        address = dev_info["address"]
        data_by_device[address]["rssi_list"].append(dev_info["rssi"])
        # 最初の出現時のmanufacturer_dataを保持
        if not data_by_device[address]["manufacturer_data"]:
             data_by_device[address]["manufacturer_data"] = dev_info["manufacturer_data"]


    # 各デバイスの平均RSSIを計算
    average_rssi_by_device: Dict[str, float] = {}
    for address, data in data_by_device.items():
        if data["rssi_list"]: # rssi_listが空でないかチェック
            average_rssi_by_device[address] = sum(data["rssi_list"]) / len(data["rssi_list"])

    # 平均RSSIで降順ソート
    sorted_avg_rssi = sorted(average_rssi_by_device.items(), key=lambda item: item[1], reverse=True)

    print("\n--- Average RSSI by Device (Sorted by Average RSSI) ---")
    if sorted_avg_rssi:
        for address, avg_rssi in sorted_avg_rssi:
            # デバイス名を取得 (最初の出現時のものを使用)
            device_name = next((dev["name"] for dev in devices_data if dev["address"] == address and dev["name"]), "-")
            print(f"{address}  Average RSSI: {avg_rssi:.2f}  Name: {device_name}")
    else:
        print("No devices found.")
    print("------------------------------------------------------")

    # CSVファイルに書き出し (ファイル名をユニークに)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"scanned_devices_avg_rssi_{timestamp}.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["address", "average_rssi", "manufacturer_data"]) # ヘッダー行

        for address, avg_rssi in sorted_avg_rssi:
            mfg_data = data_by_device[address]["manufacturer_data"]
            writer.writerow([address, avg_rssi, pretty_mfg(mfg_data)])

    print(f"\nAverage RSSI results written to {csv_file}")


if __name__ == "__main__":
    asyncio.run(main())
