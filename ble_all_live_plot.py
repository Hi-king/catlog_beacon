#!/usr/bin/env python3
"""
ble_all_devices_live_plot.py  –  全デバイスの RSSI をリアルタイム表示
  ・凡例を右側の余白に出す版
"""

import asyncio, sys, threading, time
from collections import defaultdict
from bleak import BleakScanner
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D  # ★ Import Line2D

# if sys.platform.startswith("win"): # このポリシーは古いか不要な可能性があるため削除
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

history   = defaultdict(list)            # addr -> [(t, rssi)]
max_rssi_per_device: defaultdict[str, float] = defaultdict(lambda: -float('inf')) # ★ Add type hint: addr -> max_rssi
line_dict: dict[str, Line2D] = {}        # ★ Add type hint: addr -> Line2D
colors    = list(mcolors.TABLEAU_COLORS.values())

start_t   = time.time()
lock      = threading.Lock()

def pretty_mfg(mfg_dict):
    return ",".join(f"{cid:04X}:{data.hex()}" for cid, data in mfg_dict.items()) or "-"


# ----- BLE コールバック -------------------------------------------------
negs = [
    "33F3166F",
    "88BDF924-5419-B71F-EC65-26326D79FA05",
    "8BBEB171-09AA-A360-1930-4BD6566964BB", #8BBEB171-09AA-A360-1930-4BD6566964BB  RSSI: -78  Name:-                MFG:0006:01092002638d710a78d31b4f66d6c24494a9cb5af91b277ed4bf6a  SVC:-
    # C6F43FB3-6A7B-7128-123F-6FA59EADC388  RSSI: -52  Name:-                MFG:004C:1005391c2ec6e6   SVC:-
    "C6F43FB3",
]

# Manufacturer Data (MFG) exact values to ignore
negmfg = [
    "0006:010920025a644985824fef21974cdeb711077f832d03998a80c675", # Google Fast Pair?
    "0006:01092002638d710a78d31b4f66d6c24494a9cb5af91b277ed4bf6a", # Google Fast Pair?
    "0006:0109202292840c5afb849c39b11a16c842646ae67dd4b0f57dd7bc", # Google Fast Pair?
    "00E0:01bfca3f9de8",     # Google Eddystone?
    "004C:1005391cc9f93c",   # Apple FindMy/iBeacon
    "004C:10050418b2f536",   # Apple FindMy/iBeacon
    "004C:100501102a79b4",   # Apple FindMy/iBeacon
    "004C:10055d1c5e1c90",   # Apple FindMy/iBeacon
    "004C:10052518101514",   # Apple FindMy/iBeacon
    "004C:10051c185231a5",   # Apple FindMy/iBeacon
    "004C:100505146431a6",   # Apple FindMy/iBeacon
    "004C:12020003",         # Apple Misc
    "004C:121900c586397b2388d2b6d80c7d94fe8b9c42f88318c81aa00000", # Apple Misc
    "004C:12025401",         # Apple Misc
    "004C:12020000",         # Apple Misc
    "004C:12029400",         # Apple Misc
    "004C:1006031e6a0321a9", # Apple Nearby?
    "004C:1006041d19d83e58", # Apple Nearby?
    "004C:1006001d26517848", # Apple Nearby?
    "004C:1007701faef371b818", # Apple Handoff?
    "004C:1007041bd340bc3c58", # Apple Handoff?
    "004C:09081302c0a832db1b5816080014820bb277e25b", # Apple AirPrint?
    "5701:32007000465203",   # Remo
    "0059:f303b749130e",     # Tile
    "00C4:043314121680",     # LG TV
    "0600:02fcb6a080b448a6b7ef583c63854d7f0019", # Unknown
    "0601:01061d5444dc48",   # Unknown
    "0969:e0090040f9c253031711220053", # Unknown
]

def cb(dev, adv):
    # if (dev.name is None) or (not dev.name.startswith("Cat")):
    #     return
    with lock:
        # print(dev.address, dev.rssi, dev.name, pretty_mfg(adv.manufacturer_data), adv.service_uuids)
        # if not dev.address.startswith("7BEC0100"):
        #     return
        # print(dev.name)
        
        # # 除外処理
        # if dev.name:
        #     return
        # for n in negs:
        #     if dev.address.startswith(n):
        #         return
        # # 完全一致で比較
        # mfg_str = pretty_mfg(adv.manufacturer_data)
        # if mfg_str in negmfg:
        #     return
        
        current_time = time.time() - start_t
        current_rssi = dev.rssi
        history[dev.address].append((current_time, current_rssi))
        # ★ Update max RSSI for the device ★
        max_rssi_per_device[dev.address] = max(max_rssi_per_device[dev.address], current_rssi)


async def ble_loop():
    scanner = BleakScanner(cb)
    await scanner.start()
    try:
        while True:
            await asyncio.sleep(0.02)
    finally:
        await scanner.stop()

threading.Thread(target=lambda: asyncio.run(ble_loop()), daemon=True).start()

# ----- Matplotlib -------------------------------------------------------
plt.ion()
fig, ax = plt.subplots()

# ★ 追加: 右側に凡例スペースを空ける ★
plt.subplots_adjust(right=0.75)   # 右に 25 % ほど余白を確保

ax.set_xlabel("Elapsed [s]")
ax.set_ylabel("RSSI [dBm]")
ax.set_ylim(-100, -30)
ax.set_title("BLE RSSI – All Devices (live)")

UPDATE = 0.25  # s

while plt.fignum_exists(fig.number):
    with lock:
        active_addrs = set() # Keep track of addresses plotted in this cycle
        for addr, samples in history.items():
            # ★ Check if max RSSI is above the threshold (-80 dBm) ★
            if max_rssi_per_device[addr] <= -80:
                # If the line exists, hide it
                if addr in line_dict:
                    line_dict[addr].set_visible(False)
                continue # Skip plotting this device

            active_addrs.add(addr) # Mark address as active for this cycle

            # If max RSSI > -80, proceed with plotting
            if addr not in line_dict:
                # Create the line if it doesn't exist
                # ★ Add marker='o' and markersize=3 ★
                line, = ax.plot([], [], marker='o', markersize=3, lw=1.5,
                                color=colors[len(list(filter(lambda l: l.get_visible(), line_dict.values()))) % len(colors)], # Color based on visible lines
                                label=addr)
                line_dict[addr] = line
            else:
                # Ensure the line is visible if it was previously hidden
                line_dict[addr].set_visible(True)

            # Update line data
            xs, ys = zip(*samples)
            line_dict[addr].set_data(xs, ys)

        # Update legend only with visible lines
        handles, labels = ax.get_legend_handles_labels()
        visible_handles = [h for h, l in zip(handles, labels) if l in active_addrs and line_dict[l].get_visible()]
        visible_labels = [l for l in labels if l in active_addrs and line_dict[l].get_visible()]

        if visible_handles: # Only update legend if there are visible lines
             ax.legend(visible_handles, visible_labels,
                       bbox_to_anchor=(1.02, 1), # 右上の少し外
                       loc="upper left",
                       borderaxespad=0.0,
                       fontsize="x-small")
        elif ax.get_legend() is not None:
             ax.get_legend().remove() # Remove legend if no lines are visible

        now = time.time() - start_t
        ax.set_xlim(0, max(10, now + 1))

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(UPDATE)
