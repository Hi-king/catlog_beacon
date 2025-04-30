# scan_detail.py
import asyncio
from bleak import BleakScanner

def pretty_mfg(mfg_dict):
    return ",".join(f"{cid:04X}:{data.hex()}" for cid, data in mfg_dict.items()) or "-"

devices_data = []

def cb(dev, adv):
    # 重複チェックは後で行うので、ここでは不要
    # if dev.address in seen:
    #     return
    # seen.add(dev.address)
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
    # global seen # seen は不要になる
    # seen = set()
    scanner = BleakScanner(cb)
    print("Scanning for 20 seconds...")
    await scanner.start()
    try:
        await asyncio.sleep(100)      # とりあえず 20 秒
    finally:
        await scanner.stop()
        print("Scan finished.")

    # RSSIで降順ソート
    sorted_devices = sorted(devices_data, key=lambda x: x['rssi'], reverse=True)

    # 重複排除用
    seen_addresses = set()

    print("\n--- Found Devices (Sorted by RSSI) ---")
    for dev_info in sorted_devices:
        # アドレスで重複を排除して表示
        if dev_info["address"] not in seen_addresses:
            print(
                f"{dev_info['address']}  RSSI:{dev_info['rssi']:4d}  "
                f"Name:{dev_info['name'] or '-':15s}  "
                f"MFG:{pretty_mfg(dev_info['manufacturer_data']):20s}  "
                f"SVC:{','.join(dev_info['service_uuids']) or '-'}"
            )
            seen_addresses.add(dev_info["address"])
    print("--------------------------------------")


if __name__ == "__main__":
    asyncio.run(main())
