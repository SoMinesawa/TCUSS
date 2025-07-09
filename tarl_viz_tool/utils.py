"""
Utility Functions

ログ設定、パス検証、その他の共通機能を提供
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """ログの設定
    
    Args:
        debug: デバッグレベルでログを出力するか
        log_file: ログファイルのパス（オプション）
    
    Returns:
        設定されたロガー
    """
    # ログレベルの設定
    log_level = logging.DEBUG if debug else logging.INFO
    
    # ログフォーマットの設定
    log_format = '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # ログハンドラーの設定
    handlers = []
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # ファイルハンドラー（オプション）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # ロガーの設定
    logger = logging.getLogger('tarl_viz_tool')
    logger.setLevel(log_level)
    
    # 既存のハンドラーを削除
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 新しいハンドラーを追加
    for handler in handlers:
        logger.addHandler(handler)
    
    # 親ロガーへの伝播を無効化
    logger.propagate = False
    
    return logger


def validate_paths(args: Any) -> bool:
    """パスの検証
    
    Args:
        args: コマンドライン引数
    
    Returns:
        検証結果（True: 成功, False: 失敗）
    """
    # データパスの検証
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"エラー: データパスが見つかりません: {data_path}")
        return False
    
    # シーケンスディレクトリの検証
    seq_path = data_path / args.seq
    if not seq_path.exists():
        print(f"エラー: シーケンスディレクトリが見つかりません: {seq_path}")
        return False
    
    # データファイルの存在確認（PLYまたはBIN）
    ply_files = list(seq_path.glob("*.ply"))
    bin_files = list((seq_path / "velodyne").glob("*.bin")) if (seq_path / "velodyne").exists() else []
    
    if not ply_files and not bin_files:
        print(f"エラー: データファイル（PLYまたはBIN）が見つかりません: {seq_path}")
        return False
    
    # どちらのファイル形式を使用するか表示
    if ply_files:
        print(f"PLYファイルを使用: {len(ply_files)}個のファイル")
    elif bin_files:
        print(f"BINファイルを使用: {len(bin_files)}個のファイル")
    
    # オリジナルデータパスの検証（オプション）
    if hasattr(args, 'original_data_path') and args.original_data_path:
        original_path = Path(args.original_data_path)
        if not original_path.exists():
            print(f"警告: オリジナルデータパスが見つかりません: {original_path}")
            # 警告のみで続行
    
    # パッチワークパスの検証（オプション）
    if hasattr(args, 'patchwork_path') and args.patchwork_path:
        patchwork_path = Path(args.patchwork_path)
        if not patchwork_path.exists():
            print(f"警告: パッチワークパスが見つかりません: {patchwork_path}")
            # 警告のみで続行
    
    print(f"パス検証完了: {data_path}")
    return True


def create_output_directory(output_dir: str) -> Path:
    """出力ディレクトリの作成
    
    Args:
        output_dir: 出力ディレクトリのパス
    
    Returns:
        作成されたディレクトリのPathオブジェクト
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_system_info() -> Dict[str, Any]:
    """システム情報の取得
    
    Returns:
        システム情報の辞書
    """
    import platform
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
    }
    
    # psutilが利用可能な場合のみ使用
    try:
        import psutil
        info.update({
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
        })
    except ImportError:
        import os
        info.update({
            'cpu_count': os.cpu_count() or 'unknown',
            'memory_total': 'unknown',
            'memory_available': 'unknown',
        })
    
    # GPU情報の取得（オプション）
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
        else:
            info['gpu_count'] = 0
    except ImportError:
        info['gpu_count'] = 0
    
    return info


def format_size(size_bytes: int) -> str:
    """バイトサイズを人間が読みやすい形式に変換
    
    Args:
        size_bytes: バイト数
    
    Returns:
        フォーマットされたサイズ文字列
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f}{size_names[i]}"


def format_time(seconds: float) -> str:
    """秒数を時分秒の形式に変換
    
    Args:
        seconds: 秒数
    
    Returns:
        フォーマットされた時間文字列
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.2f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}時間{minutes}分{secs:.2f}秒"


def print_banner():
    """バナーメッセージの表示"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                               ║
    ║                    TARL Clustering Visualization Tool                         ║
    ║                                                                               ║
    ║                   TARLクラスタリング ハイパーパラメータ最適化ツール                 ║
    ║                                                                               ║
    ║                                 Version 1.0.0                                 ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_system_info():
    """システム情報の表示"""
    print("\n=== システム情報 ===")
    info = get_system_info()
    
    print(f"プラットフォーム: {info['platform']}")
    print(f"Python バージョン: {info['python_version']}")
    print(f"CPU コア数: {info['cpu_count']}")
    print(f"メモリ総量: {format_size(info['memory_total'])}")
    print(f"メモリ使用可能: {format_size(info['memory_available'])}")
    
    if info['gpu_count'] > 0:
        print(f"GPU数: {info['gpu_count']}")
        print(f"GPU名: {info['gpu_name']}")
        print(f"GPU メモリ: {format_size(info['gpu_memory'])}")
    else:
        print("GPU: 検出されませんでした")
    
    print()


def check_dependencies():
    """依存関係の確認
    
    Returns:
        確認結果（True: 成功, False: 失敗）
    """
    # 必須パッケージ
    required_packages = [
        'numpy',
        'open3d',
        'MinkowskiEngine',
        'sklearn',
        'matplotlib'
    ]
    
    # オプションパッケージ
    optional_packages = [
        'hdbscan',
        'cuml',
        'psutil'
    ]
    
    missing_required = []
    missing_optional = []
    
    # 必須パッケージの確認
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    # オプションパッケージの確認
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    # 結果の表示
    if missing_required:
        print(f"エラー: 以下の必須パッケージがインストールされていません: {', '.join(missing_required)}")
        print("pip install <package_name> でインストールしてください")
        return False
    
    if missing_optional:
        print(f"警告: 以下のオプションパッケージが見つかりません: {', '.join(missing_optional)}")
        print("一部の機能が制限される場合があります")
    
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全な除算（ゼロ除算を回避）
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 分母がゼロの場合のデフォルト値
    
    Returns:
        除算結果
    """
    return numerator / denominator if denominator != 0 else default


def dict_to_string(d: Dict[str, Any], indent: int = 0) -> str:
    """辞書を文字列に変換（階層構造を保持）
    
    Args:
        d: 辞書
        indent: インデントレベル
    
    Returns:
        フォーマットされた文字列
    """
    lines = []
    for key, value in d.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(dict_to_string(value, indent + 1))
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def calculate_memory_usage(data_size: int, dtype_size: int = 4) -> str:
    """メモリ使用量の計算
    
    Args:
        data_size: データサイズ（要素数）
        dtype_size: データ型のサイズ（バイト）
    
    Returns:
        メモリ使用量の文字列
    """
    memory_bytes = data_size * dtype_size
    return format_size(memory_bytes)


def validate_numeric_range(value: float, min_val: float, max_val: float, name: str) -> bool:
    """数値の範囲検証
    
    Args:
        value: 検証する値
        min_val: 最小値
        max_val: 最大値
        name: パラメータ名
    
    Returns:
        検証結果
    """
    if not (min_val <= value <= max_val):
        print(f"エラー: {name}は{min_val}から{max_val}の範囲内である必要があります（現在値: {value}）")
        return False
    return True 