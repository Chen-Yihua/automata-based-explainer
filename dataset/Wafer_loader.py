import os
import zipfile
import urllib.request
import random
from typing import List, Sequence, Tuple, Optional

import numpy as np


WAFER_URLS = [
    "https://www.timeseriesclassification.com/aeon-toolkit/Wafer.zip",
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

Wafer_SYMBOLS = ['VL', 'L', 'SL', 'M', 'SH', 'H', 'VH']


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


def _download_wafer_dataset(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    try:
        _find_dataset_files(data_dir, "Wafer")
        return
    except FileNotFoundError:
        pass

    zip_path = os.path.join(data_dir, "Wafer.zip")
    for url in WAFER_URLS:
        if _download_file(url, zip_path):
            break
    else:
        raise FileNotFoundError(
            "Could not download Wafer automatically. Please place Wafer_TRAIN.txt and Wafer_TEST.txt under datasets/Wafer/."
        )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
    except zipfile.BadZipFile as e:
        raise FileNotFoundError("Downloaded Wafer zip is invalid.") from e

    _find_dataset_files(data_dir, "Wafer")


def _pad_or_truncate_symbol_sequence(
    seq: Sequence[str], target_len: Optional[int], pad_symbol: str = "<PAD>"
) -> List[str]:
    seq = list(seq)
    if target_len is None:
        return seq
    if len(seq) > target_len:
        return seq[:target_len]
    if len(seq) < target_len:
        return seq + [pad_symbol] * (target_len - len(seq))
    return seq


def _choose_n_segments(
    seq_len: int,
    n_segments: Optional[int],
    base_segments: int,
    segment_radius: int,
    random_segment_length: bool,
    rng: random.Random,
) -> int:
    """
    Choose the output PAA length for a single sequence.

    - If random_segment_length=False:
        use n_segments if provided, otherwise use base_segments.
    - If random_segment_length=True:
        sample uniformly from [base_segments - segment_radius, base_segments + segment_radius].
    """
    if not random_segment_length:
        target = n_segments if n_segments is not None else base_segments
        return max(1, min(int(target), seq_len))

    low = max(1, int(base_segments - segment_radius))
    high = min(seq_len, int(base_segments + segment_radius))
    if low > high:
        low = high
    return rng.randint(low, high)


def load_Wafer_sequences(
    data_dir: str = "datasets/Wafer",
    alphabet_size: int = 7,
    discretize_method: str = "quantile",  # "quantile" | "sax"
    use_paa: bool = True,
    n_segments: Optional[int] = None,
    base_segments: int = 20,
    segment_radius: int = 5,
    random_segment_length: bool = False,
    compress: bool = False,
    normalize_per_sequence: bool = True,
    pad_to_length: Optional[int] = None,
    return_raw: bool = False,
    random_state: int = 42,
):
    """
    Load Wafer and convert each real-valued time series into a symbolic sequence.

    Recommended variable-length setting:
        use_paa=True,
        base_segments=20,
        segment_radius=5,
        random_segment_length=True

    Then each sequence is compressed into a random length within [15, 25].
    """
    if alphabet_size > len(Wafer_SYMBOLS):
        raise ValueError(f"alphabet_size should be <= {len(Wafer_SYMBOLS)} for Wafer_SYMBOLS")
    if discretize_method not in {"quantile", "sax"}:
        raise ValueError("discretize_method must be 'quantile' or 'sax'")
    if use_paa:
        if random_segment_length:
            if base_segments <= 0:
                raise ValueError("base_segments must be positive")
            if segment_radius < 0:
                raise ValueError("segment_radius must be >= 0")
        else:
            if n_segments is None and base_segments is None:
                raise ValueError("Please provide n_segments or base_segments")
            target = n_segments if n_segments is not None else base_segments
            if target <= 0:
                raise ValueError("n_segments/base_segments must be positive")

    _download_wafer_dataset(data_dir)
    train_path, test_path = _find_dataset_files(data_dir, "Wafer")

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
            seg_len = _choose_n_segments(
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

    symbols = list(Wafer_SYMBOLS[:alphabet_size])
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
        pad_symbol = f"{Wafer_SYMBOLS[-1]}_PAD"
        X_train_sym = [_pad_or_truncate_symbol_sequence(seq, pad_to_length, pad_symbol) for seq in X_train_sym]
        X_test_sym = [_pad_or_truncate_symbol_sequence(seq, pad_to_length, pad_symbol) for seq in X_test_sym]

    if return_raw:
        return X_train_sym, X_test_sym, y_train, y_test, X_train_raw, X_test_raw
    return X_train_sym, X_test_sym, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_Wafer_sequences(
        alphabet_size=7,
        discretize_method="quantile",
        use_paa=True,
        base_segments=15,
        segment_radius=5,
        random_segment_length=True,
        compress=False,
        random_state=42,
    )
    print(f"Loaded Wafer: train={len(X_train)}, test={len(X_test)}")
    print(f"Example sequence: {X_train[0]}")
    print(f"Example length: {len(X_train[0])}")
