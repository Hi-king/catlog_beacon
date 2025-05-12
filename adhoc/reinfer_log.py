import argparse
import pandas as pd
import sys
import numpy as np # numpyをインポート

# --- ライブラリからのインポート ---
from blelocation.model_utils import load_model_bundle
from blelocation.feature_engineering import extract_inference_features

# --- Parameters ---
MODEL_BUNDLE_PATH = "location_classification_bundle.joblib" # 学習済みモデルバンドル
# 推論に必要な最小データポイント数はモデルバンドルから取得

def main(input_log_path, output_log_path):
    print(f"入力ログファイル: {input_log_path} を読み込みます。")

    try:
        df = pd.read_csv(input_log_path)
        # 必要に応じてカラム名をリネーム
        if "datetime" in df.columns and "predicted_locatio" in df.columns:
             df = df.rename(columns={"datetime": "timestamp", "predicted_locatio": "location"})
        elif "timestamp" not in df.columns or "rssi_dbm" not in df.columns:
             print(f"エラー: ログファイル '{input_log_path}' に必要な 'timestamp' または 'rssi_dbm' カラムが見つかりません。")
             sys.exit(1)

        # timestampをdatetimeに変換し、ソート
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    except FileNotFoundError:
        print(f"エラー: 入力ログファイル '{input_log_path}' が見つかりません。")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: 入力ログファイル '{input_log_path}' の読み込み中にエラーが発生しました: {e}")
        sys.exit(1)

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

    # --- 推論の実行 ---
    # ログデータ全体に対してウィンドウ処理で推論を実行
    window_size_sec = 60 # 例: 60秒のウィンドウで推論
    step_size_sec = 10 # 例: 10秒ごとにウィンドウをスライド

    inferred_data = [] # (timestamp, predicted_location, probability) のリスト

    # データの開始時刻と終了時刻を取得
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()

    current_window_end = start_time + pd.Timedelta(seconds=window_size_sec)

    while current_window_end <= end_time + pd.Timedelta(seconds=step_size_sec): # 最後のデータまで処理できるように少し余裕を持たせる
        window_start_time = current_window_end - pd.Timedelta(seconds=window_size_sec)

        # 現在のウィンドウ内のデータを抽出
        window_df = df[(df['timestamp'] >= window_start_time) & (df['timestamp'] < current_window_end)].copy()

        if len(window_df) >= min_points_per_window:
            # ウィンドウ内の最終タイムスタンプをこの推論の代表タイムスタンプとする
            inference_timestamp = window_df['timestamp'].max()

            feature_df = extract_inference_features(
                data=window_df,
                rssi_col='rssi_dbm',
                feature_names=feature_names,
                min_points_required=min_points_per_window
            )

            if feature_df is not None:
                try:
                    features_scaled = scaler.transform(feature_df)
                    probabilities = model.predict_proba(features_scaled)[0]

                    # 上位2つの予測を取得
                    sorted_indices = np.argsort(probabilities)[::-1] # 降順ソートされたインデックス

                    predicted_index = sorted_indices[0]
                    predicted_location = class_names[predicted_index]
                    predicted_probability = probabilities[predicted_index]

                    second_predicted_location = None
                    second_predicted_probability = None
                    if len(class_names) > 1 and len(sorted_indices) > 1:
                        second_predicted_index = sorted_indices[1]
                        second_predicted_location = class_names[second_predicted_index]
                        second_predicted_probability = probabilities[second_predicted_index]

                    inferred_data.append({
                        'timestamp': inference_timestamp,
                        'predicted_location': predicted_location,
                        'predicted_probability': predicted_probability,
                        'second_predicted_location': second_predicted_location,
                        'second_predicted_probability': second_predicted_probability
                    })

                except Exception as e:
                    print(f"警告: タイムスタンプ {inference_timestamp} 付近の推論中にエラーが発生しました: {e}")
                    # エラーが発生した場合もデータを記録しない
                    pass # エラー時はスキップ
            else:
                 # データ不足の場合もデータを記録しない
                 pass # データ不足時はスキップ
        else:
            # データポイントが足りない場合もデータを記録しない
            pass # データ不足時はスキップ


        # ウィンドウをスライド
        current_window_end += pd.Timedelta(seconds=step_size_sec)

    if not inferred_data:
        print("警告: 推論結果が得られませんでした。出力ファイルは作成されません。")
        sys.exit(0)

    # 推論結果をDataFrameに変換
    inferred_df = pd.DataFrame(inferred_data)

    # 元のデータフレームと推論結果をマージ（タイムスタンプで近いものを結合）
    # ここでは、推論結果のタイムスタンプに最も近い元のデータの行に推論結果を結合することを考えます。
    # より洗練された方法としては、推論結果を一定間隔で補間し、元のデータフレームにマージすることも可能です。
    # 簡略化のため、ここでは推論結果のDataFrameを作成し、必要に応じて元のデータと結合せずに独立したファイルとして出力します。
    # もし元のデータフレームに推論結果を追加したい場合は、マージ処理を実装する必要があります。
    # 今回は、推論結果のみを含む新しいCSVファイルを作成します。

    # --- 推論結果の出力 ---
    print(f"推論結果を '{output_log_path}' に出力します。")
    try:
        inferred_df.to_csv(output_log_path, index=False)
        print("出力が完了しました。")
    except Exception as e:
        print(f"エラー: 出力ファイル '{output_log_path}' の書き込み中にエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Re-infer location from log file and output new log file.')
    parser.add_argument('input_log', help='Path to the input log file (CSV).')
    parser.add_argument('output_log', help='Path to the output log file (CSV).')
    args = parser.parse_args()

    main(args.input_log, args.output_log)
