import pandas as pd
import numpy as np
from typing import List, Optional, Dict # 型ヒント用

def extract_inference_features(
    data: pd.DataFrame,
    rssi_col: str = 'rssi_dbm',
    feature_names: Optional[List[str]] = None,
    min_points_required: int = 1
) -> Optional[pd.DataFrame]:
    """
    指定されたRSSIデータから推論用の特徴量を抽出する汎用関数。

    Args:
        data (pd.DataFrame): RSSIデータを含むDataFrame。RSSIデータ列が必要。
        rssi_col (str): RSSIデータが含まれる列の名前。デフォルトは 'rssi_dbm'。
        feature_names (Optional[List[str]]): 抽出したい特徴量の名前のリスト。
            Noneの場合はデフォルトの特徴量セットを使用。
            デフォルト: ['rssi_mean', 'rssi_std', 'rssi_min', 'rssi_max', 'rssi_median', 'rssi_range']
        min_points_required (int): 特徴量抽出に必要な最小データポイント数。デフォルトは 1。

    Returns:
        Optional[pd.DataFrame]: 抽出された特徴量を含むDataFrame (1行)。
                                データポイント数が不足している場合はNoneを返す。
    """
    if not isinstance(data, pd.DataFrame) or data.empty or rssi_col not in data.columns:
        # print("[DEBUG] Input data is not a valid DataFrame or rssi_col is missing.")
        return None

    if len(data) < min_points_required:
        # print(f"[DEBUG] Not enough data points: {len(data)} < {min_points_required}")
        return None

    rssi_series = data[rssi_col].astype(float) # 念のためfloatに変換

    # デフォルトの特徴量リスト
    default_features = [
        'rssi_mean', 'rssi_std', 'rssi_min', 'rssi_max', 'rssi_median', 'rssi_range'
    ]
    if feature_names is None:
        feature_names = default_features

    # 計算可能な特徴量を格納する辞書
    calculated_features: Dict[str, Optional[float]] = {}

    # 各特徴量を計算
    if 'rssi_mean' in feature_names:
        calculated_features['rssi_mean'] = rssi_series.mean()
    if 'rssi_std' in feature_names:
        # データポイントが1つの場合、stdは0とする
        calculated_features['rssi_std'] = rssi_series.std() if len(rssi_series) > 1 else 0.0
    if 'rssi_min' in feature_names:
        calculated_features['rssi_min'] = rssi_series.min()
    if 'rssi_max' in feature_names:
        calculated_features['rssi_max'] = rssi_series.max()
    if 'rssi_median' in feature_names:
        calculated_features['rssi_median'] = rssi_series.median()
    if 'rssi_range' in feature_names and 'rssi_min' in calculated_features and 'rssi_max' in calculated_features:
         # min/maxが計算済みの場合のみ計算
        if calculated_features['rssi_min'] is not None and calculated_features['rssi_max'] is not None:
             calculated_features['rssi_range'] = calculated_features['rssi_max'] - calculated_features['rssi_min']
        else:
             calculated_features['rssi_range'] = None # 計算不可

    # Noneが含まれる特徴量を除外 (エラー発生時など)
    final_features = {k: v for k, v in calculated_features.items() if v is not None}

    # 要求された特徴量名でDataFrameを作成
    # 存在しない特徴量名が要求された場合、その列は作成されない
    feature_df = pd.DataFrame([final_features], columns=feature_names)

    # 標準偏差はNoneはない
    if 'rssi_std' in feature_df.columns:
        assert feature_df['rssi_std'].notna().all()

    # 要求されたすべての特徴量が含まれているか確認 
    missing_features = [f for f in feature_names if f not in feature_df.columns]
    assert not missing_features

    return feature_df
