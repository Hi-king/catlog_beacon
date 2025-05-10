#!/usr/bin/env python3
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

def create_training_data(input_csv_path: str, start_time_str: str, room_name: str, end_time_str: Optional[str] = None, output_dir: str = "data"):
    """
    指定されたCSVファイルから時間範囲でデータを抽出し、新しいCSVファイルとして保存する。

    Args:
        input_csv_path (str): 入力CSVファイルのパス。
        start_time_str (str): 開始時間 (YYYYMMDDHHMM形式)。
        room_name (str): 部屋名。
        end_time_str (Optional[str]): 終了時間 (YYYYMMDDHHMM形式)。省略時は現在時刻。
        output_dir (str): 出力ディレクトリ名。
    """
    try:
        start_dt = datetime.strptime(start_time_str, "%Y%m%d%H%M")
        if end_time_str is None:
            end_dt = datetime.now()
            print(f"情報: 終了時間が指定されていないため、現在時刻 ({end_dt.strftime('%Y%m%d%H%M')}) を使用します。")
        else:
            end_dt = datetime.strptime(end_time_str, "%Y%m%d%H%M")

        if start_dt > end_dt:
            print(f"エラー: 開始時間 ({start_dt.strftime('%Y%m%d%H%M')}) が終了時間 ({end_dt.strftime('%Y%m%d%H%M')}) より後になっています。")
            return

    except ValueError:
        print("エラー: 時間の形式が正しくありません。YYYYMMDDHHMM形式で指定してください。")
        return

    input_file = Path(input_csv_path)
    if not input_file.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_csv_path}")
        return

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}")
        return

    if "datetime" not in df.columns:
        print("エラー: CSVファイルに 'datetime' 列が見つかりません。")
        return

    # 'datetime' 列をdatetimeオブジェクトに変換 (エラーは無視してNaTにする)
    # 元のデータ形式が '%Y-%m-%d %H:%M:%S.%f' のようなので、それに合わせる
    # もし形式が異なる場合は、適切なフォーマット文字列を指定する必要がある
    try:
        df["datetime_dt"] = pd.to_datetime(df["datetime"], errors='coerce')
    except Exception as e:
        print(f"エラー: 'datetime' 列の日時変換に失敗しました: {e}")
        print("ヒント: 'datetime'列の形式が '%Y-%m-%d %H:%M:%S.%f' またはpandasが解釈可能な形式であることを確認してください。")
        return

    # NaTになった行を除外 (変換できなかった行)
    df_filtered_by_time = df.dropna(subset=["datetime_dt"])

    # 時間範囲でフィルタリング
    df_filtered_by_time = df_filtered_by_time[
        (df_filtered_by_time["datetime_dt"] >= start_dt) &
        (df_filtered_by_time["datetime_dt"] <= end_dt)
    ]

    if df_filtered_by_time.empty:
        print("指定された時間範囲に該当するデータが見つかりませんでした。")
        return

    # "datetime_dt" 列は不要なので削除
    df_to_save = df_filtered_by_time.drop(columns=["datetime_dt"])

    # 出力ディレクトリを作成 (存在しない場合)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 出力ファイル名を作成
    # ファイル名のタイムスタンプ部分は引数の開始時間を使用
    output_filename = f"{start_dt.strftime('%Y%m%d%H%M')}_{room_name}.csv"
    output_file_path = output_path / output_filename

    try:
        df_to_save.to_csv(output_file_path, index=False)
        print(f"教師データを作成しました: {output_file_path}")
    except Exception as e:
        print(f"エラー: CSVファイルへの書き込みに失敗しました: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSSIログから教師データを作成します。")
    parser.add_argument("start_time", help="開始時間 (YYYYMMDDHHMM形式)")
    parser.add_argument("room_name", help="部屋名")
    parser.add_argument(
        "end_time",
        nargs="?", # 0または1個の引数を許可
        default=None, # デフォルトはNoneとし、関数側で現在時刻を処理
        help="終了時間 (YYYYMMDDHHMM形式)。省略した場合は現在時刻が使用されます。"
    )
    parser.add_argument(
        "--input_csv",
        default="/Users/keisuke.ogaki/ghq/github.com/Hi-king/catlog_beacon/rssi_inference_log.csv",
        help="入力CSVファイルのパス (デフォルト: /Users/keisuke.ogaki/ghq/github.com/Hi-king/catlog_beacon/rssi_inference_log.csv)"
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="出力ディレクトリ (デフォルト: data)"
    )

    args = parser.parse_args()

    create_training_data(
        input_csv_path=args.input_csv,
        start_time_str=args.start_time,
        room_name=args.room_name,
        end_time_str=args.end_time, # Noneの場合あり
        output_dir=args.output_dir
    )
