import joblib
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_BUNDLE_FILENAME = "location_classification_bundle.joblib"

def load_model_bundle(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    指定されたパスからモデルバンドル（.joblibファイル）を読み込む。

    Args:
        path (Optional[str]): モデルバンドルファイルのパス。Noneの場合はデフォルトのファイル名を使用。

    Returns:
        Optional[Dict[str, Any]]: 読み込んだバンドルの内容を含む辞書。
                                   ファイルが存在しない場合や読み込みに失敗した場合はNoneを返す。
                                   期待されるキー: 'model', 'scaler', 'classes', 'features', 'min_points_per_window'
    """
    if path is None:
        path = DEFAULT_BUNDLE_FILENAME

    bundle_path = Path(path)
    if not bundle_path.exists():
        print(f"[ERROR] Model bundle not found at: {bundle_path}")
        return None
    try:
        bundle = joblib.load(bundle_path)
        print(f"[INFO] Model bundle loaded successfully from {bundle_path}")
        # 必要なキーが存在するか簡単なチェック（オプション）
        required_keys = ['model', 'scaler', 'classes', 'features', 'min_points_per_window']
        missing_keys = [key for key in required_keys if key not in bundle]
        if missing_keys:
            print(f"[WARN] The following keys are missing in the loaded bundle: {missing_keys}")
        return bundle
    except Exception as e:
        print(f"[ERROR] Failed to load model bundle from {bundle_path}: {e}")
        return None

def save_model_bundle(bundle_data: Dict[str, Any], path: Optional[str] = None) -> bool:
    """
    モデルバンドルデータ（辞書）を指定されたパスに保存する。

    Args:
        bundle_data (Dict[str, Any]): 保存するバンドルデータを含む辞書。
                                      期待されるキー: 'model', 'scaler', 'classes', 'features', 'min_points_per_window'
        path (Optional[str]): 保存先のファイルパス。Noneの場合はデフォルトのファイル名を使用。

    Returns:
        bool: 保存に成功した場合はTrue、失敗した場合はFalse。
    """
    if path is None:
        path = DEFAULT_BUNDLE_FILENAME

    bundle_path = Path(path)
    try:
        # 保存前にディレクトリが存在するか確認し、なければ作成（オプション）
        # bundle_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle_data, bundle_path)
        print(f"[INFO] Model bundle saved successfully to {bundle_path}")
        # 保存内容の簡単な表示（オプション）
        print(f"  Includes keys: {list(bundle_data.keys())}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save model bundle to {bundle_path}: {e}")
        return False
