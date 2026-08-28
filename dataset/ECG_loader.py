
import os
import zipfile
import urllib.request
import random
from typing import List, Sequence, Tuple, Optional

import numpy as np


ECG5000_URLS = [
    "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip",
]

SAX_BREAKPOINTS = {
    3: [-0.43, 0.43],
    4: [-0.67, 0.0, 0.67],
    5: [-0.84, -0.25, 0.25, 0.84],
    6: [-0.97, -0.43, 0.0, 0.43, 0.97],
    7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
    8: [-1.15, -0.67, -0.32, 0.0, 0.32, 0.67, 1.15],
    9: [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
    10: [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28],
}

ECG_SYMBOLS = ['VL', 'L', 'SL', 'M', 'SH', 'H', 'VH']


def compress_symbol_sequence(seq: Sequence[str]) -> List[str]:
    if not seq:
        return list(seq)
    compressed = [seq[0]]
    for s in seq[1:]:
        if s != compressed[-1]:
            compressed.append(s)
    return compressed


def z_normalize(seq: np.ndarray) -> np.ndarray:
    mean = np.mean(seq)
    std = np.std(seq)
    if std == 0:
        return np.zeros_like(seq)
    return (seq - mean) / std


def paa(seq: np.ndarray, n_segments: int) -> np.ndarray:
    """Piecewise Aggregate Approximation. Output length == n_segments."""
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    if n_segments == len(seq):
        return seq.astype(np.float32)
    if n_segments > len(seq):
        idx = np.linspace(0, len(seq) - 1, n_segments)
        idx = np.round(idx).astype(int)
        return seq[idx].astype(np.float32)

    idx = np.linspace(0, len(seq), n_segments + 1)
    result = []
    for i in range(n_segments):
        start = int(round(idx[i]))
        end = int(round(idx[i + 1]))
        segment = seq[start:end]
        if len(segment) == 0:
            nearest = min(max(start, 0), len(seq) - 1)
            result.append(float(seq[nearest]))
        else:
            result.append(float(np.mean(segment)))
    return np.asarray(result, dtype=np.float32)


def fit_quantile_breakpoints(train_values: np.ndarray, alphabet_size: int) -> np.ndarray:
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least 2")
    quantiles = np.linspace(0, 1, alphabet_size + 1)[1:-1]
    return np.quantile(train_values, quantiles)


def symbolize_with_breakpoints(seq: np.ndarray, breakpoints: np.ndarray, symbols: Sequence[str]) -> List[str]:
    idxs = np.searchsorted(breakpoints, seq, side="right")
    return [symbols[i] for i in idxs]


def _load_ucr_txt_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.float32)
    y = data[:, 0].astype(np.int64)
    X = data[:, 1:].astype(np.float32)
    return X, y


def _download_file(url: str, output_path: str) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(output_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception:
        return False


def _find_dataset_files(data_dir: str, dataset_name: str) -> Tuple[str, str]:
    train_candidates = [
        os.path.join(data_dir, f"{dataset_name}_TRAIN.txt"),
        os.path.join(data_dir, f"{dataset_name}_TRAIN.tsv"),
        os.path.join(data_dir, f"{dataset_name}_TRAIN"),
    ]
    test_candidates = [
        os.path.join(data_dir, f"{dataset_name}_TEST.txt"),
        os.path.join(data_dir, f"{dataset_name}_TEST.tsv"),
        os.path.join(data_dir, f"{dataset_name}_TEST"),
    ]

    train_path = next((p for p in train_candidates if os.path.exists(p)), None)
    test_path = next((p for p in test_candidates if os.path.exists(p)), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError(f"Could not locate {dataset_name} train/test files under {data_dir}")
    return train_path, test_path


def _download_ecg5000_dataset(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    try:
        _find_dataset_files(data_dir, "ECG5000")
        return
    except FileNotFoundError:
        pass

    zip_path = os.path.join(data_dir, "ECG5000.zip")
    if not _download_file(ECG5000_URLS[0], zip_path):
        raise FileNotFoundError(
            "Could not download ECG5000 automatically. Please place ECG5000_TRAIN.txt and ECG5000_TEST.txt under datasets/ECG5000/."
        )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
    except zipfile.BadZipFile as e:
        raise FileNotFoundError("Downloaded ECG5000 zip is invalid.") from e

    _find_dataset_files(data_dir, "ECG5000")


def _pad_or_truncate_symbol_sequence(seq: Sequence[str], target_len: Optional[int], pad_symbol: str = "<PAD>") -> List[str]:
    seq = list(seq)
    if target_len is None:
        return seq
    if len(seq) > target_len:
        return seq[:target_len]
    if len(seq) < target_len:
        return seq + [pad_symbol] * (target_len - len(seq))
    return seq


def _choose_segment_count(
    seq_len: int,
    n_segments: Optional[int] = None,
    base_segments: int = 20,
    segment_radius: int = 0,
    random_segment_length: bool = False,
    rng: Optional[random.Random] = None,
) -> int:
    """
    決定單一序列要壓成幾段。

    - random_segment_length=False:
        使用 n_segments；若 n_segments 為 None，則使用 base_segments
    - random_segment_length=True:
        從 [base_segments - segment_radius, base_segments + segment_radius] 中隨機選
    """
    if seq_len <= 0:
        return 1

    if not random_segment_length:
        target = n_segments if n_segments is not None else base_segments
        return max(1, min(int(target), seq_len))

    if rng is None:
        rng = random.Random()

    low = max(1, base_segments - segment_radius)
    high = max(low, base_segments + segment_radius)

    high = min(high, seq_len)
    low = min(low, high)
    return rng.randint(low, high)


def load_ECG5000_sequences(
    data_dir: str = "datasets/ECG5000",
    alphabet_size: int = 7,
    discretize_method: str = "quantile",  # "quantile" | "sax"
    use_paa: bool = True,
    n_segments: Optional[int] = None,
    base_segments: int = 20,
    segment_radius: int = 0,
    random_segment_length: bool = False,
    compress: bool = False,
    normalize_per_sequence: bool = True,
    pad_to_length: Optional[int] = None,
    return_raw: bool = False,
    random_state: int = 42,
):
    """
    Load ECG5000 and convert each real-valued time series into a symbolic sequence.

    若 random_segment_length=True，則每條序列會被壓成
    [base_segments - segment_radius, base_segments + segment_radius] 範圍內的隨機長度。
    """
    if alphabet_size > 10:
        raise ValueError("alphabet_size should be <= 10 for your DFA setting")
    if discretize_method not in {"quantile", "sax"}:
        raise ValueError("discretize_method must be 'quantile' or 'sax'")
    if segment_radius < 0:
        raise ValueError("segment_radius must be >= 0")
    if base_segments <= 0:
        raise ValueError("base_segments must be > 0")
    if n_segments is not None and n_segments <= 0:
        raise ValueError("n_segments must be positive when provided")

    _download_ecg5000_dataset(data_dir)
    train_path, test_path = _find_dataset_files(data_dir, "ECG5000")

    X_train_raw, y_train = _load_ucr_txt_file(train_path)
    X_test_raw, y_test = _load_ucr_txt_file(test_path)

    unique_labels = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.asarray([label_map[v] for v in y_train], dtype=np.int64)
    y_test = np.asarray([label_map[v] for v in y_test], dtype=np.int64)

    rng = random.Random(random_state)

    def _prepare_sequence(seq: np.ndarray) -> np.ndarray:
        seq_out = z_normalize(seq) if normalize_per_sequence else seq.astype(np.float32)
        if use_paa:
            seg_len = _choose_segment_count(
                seq_len=len(seq_out),
                n_segments=n_segments,
                base_segments=base_segments,
                segment_radius=segment_radius,
                random_segment_length=random_segment_length,
                rng=rng,
            )
            seq_out = paa(seq_out, n_segments=seg_len)
        return seq_out.astype(np.float32)

    X_train_proc = [_prepare_sequence(seq) for seq in X_train_raw]
    X_test_proc = [_prepare_sequence(seq) for seq in X_test_raw]

    if alphabet_size != len(ECG_SYMBOLS):
        raise ValueError(
            f"ECG symbol set requires alphabet_size={len(ECG_SYMBOLS)}, got {alphabet_size}"
        )
    symbols = ECG_SYMBOLS
    if discretize_method == "quantile":
        train_values = np.concatenate(X_train_proc)
        breakpoints = fit_quantile_breakpoints(train_values, alphabet_size)
    else:
        if alphabet_size not in SAX_BREAKPOINTS:
            raise ValueError(f"SAX breakpoints for alphabet_size={alphabet_size} are not defined")
        breakpoints = np.asarray(SAX_BREAKPOINTS[alphabet_size], dtype=np.float32)

    X_train_sym = [symbolize_with_breakpoints(seq, breakpoints, symbols) for seq in X_train_proc]
    X_test_sym = [symbolize_with_breakpoints(seq, breakpoints, symbols) for seq in X_test_proc]

    if compress:
        X_train_sym = [compress_symbol_sequence(seq) for seq in X_train_sym]
        X_test_sym = [compress_symbol_sequence(seq) for seq in X_test_sym]

    if pad_to_length is not None:
        pad_symbol = "<PAD>"
        X_train_sym = [_pad_or_truncate_symbol_sequence(seq, pad_to_length, pad_symbol) for seq in X_train_sym]
        X_test_sym = [_pad_or_truncate_symbol_sequence(seq, pad_to_length, pad_symbol) for seq in X_test_sym]

    if return_raw:
        return X_train_sym, X_test_sym, y_train, y_test, X_train_raw, X_test_raw
    return X_train_sym, X_test_sym, y_train, y_test


def load_EGG_sequences(*args, **kwargs):
    """Alias kept for compatibility with your requested filename naming."""
    return load_ECG5000_sequences(*args, **kwargs)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_ECG5000_sequences(
        alphabet_size=7,
        discretize_method="quantile",
        use_paa=True,
        base_segments=20,
        segment_radius=5,
        random_segment_length=True,
        compress=False,
        random_state=42,
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Example symbolic sequence: {X_train[0]}")
    print(f"Example sequence length: {len(X_train[0])}")
    print(f"Example label: {y_train[0]}")
