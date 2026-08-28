import math
import pickle
import random
from sklearn.model_selection import train_test_split

DIRECTIONS_4 = ['R', 'U', 'L', 'D']


def compress_stroke_sequence(seq):
    """Merge consecutive identical symbols."""
    if not seq:
        return seq
    compressed = [seq[0]]
    for s in seq[1:]:
        if s != compressed[-1]:
            compressed.append(s)
    return compressed


def discretize_4dir(dx, dy):
    """Map a non-zero displacement to one of 4 directions using dominant axis."""
    if dx == 0 and dy == 0:
        raise ValueError("Zero displacement cannot be mapped to the 4-direction alphabet.")

    if abs(dx) >= abs(dy):
        return 'R' if dx >= 0 else 'L'
    else:
        return 'D' if dy >= 0 else 'U'


def _choose_segment_count(
    seq_len,
    n_segments=None,
    base_segments=40,
    segment_radius=0,
    random_segment_length=False,
    rng=None,
):
    """Choose how many segments to use for a sequence."""
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


def _first_nonzero_direction(points):
    """Find the first non-zero displacement in a list of points and map it to 4 directions."""
    for pt in points:
        dx, dy = pt[0], pt[1]
        if dx != 0 or dy != 0:
            return discretize_4dir(dx, dy)
    return None


def load_mnist_stroke_sequences(
    data_path="datasets/mnist-digits-as-stroke-sequences/mnist_strokes.pkl",
    allow_segment=True,
    n_segments=40,
    base_segments=40,
    segment_radius=0,
    random_segment_length=False,
    compress_repeats=False,
    random_state=42,
):
    """
    Load MNIST stroke data and convert each sequence to a 4-direction symbolic sequence.

    The output alphabet is exactly:
        {R, U, L, D}

    No raw (dx, dy) tuples and no zero-motion symbol are produced.
    """
    if segment_radius < 0:
        raise ValueError("segment_radius must be >= 0")
    if base_segments <= 0:
        raise ValueError("base_segments must be > 0")
    if n_segments is not None and n_segments <= 0:
        raise ValueError("n_segments must be > 0 when provided")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    X, y = [], []
    rng = random.Random(random_state)

    def segment_average_direction(seq, n_segments=10):
        """
        Segment a sequence into n_segments blocks.
        Each block is summarized by the average displacement, then mapped to 1 of 4 directions.
        Zero-mean segments inherit the previous direction when possible; otherwise they use the
        first future non-zero direction. As a last resort, they default to 'R'.
        """
        L = len(seq)
        indices = [int(round(i * L / n_segments)) for i in range(n_segments + 1)]
        result_syms = []
        prev_sym = None

        for i in range(n_segments):
            seg = seq[indices[i]:indices[i + 1]]

            if not seg:
                if prev_sym is not None:
                    result_syms.append(prev_sym)
                    continue

                fallback = None
                for j in range(i + 1, n_segments):
                    future_seg = seq[indices[j]:indices[j + 1]]
                    fallback = _first_nonzero_direction(future_seg)
                    if fallback is not None:
                        break

                if fallback is None:
                    fallback = 'R'

                result_syms.append(fallback)
                prev_sym = fallback
                continue

            dx_sum = sum(pt[0] for pt in seg)
            dy_sum = sum(pt[1] for pt in seg)
            avg_dx = round(dx_sum / len(seg))
            avg_dy = round(dy_sum / len(seg))

            if avg_dx == 0 and avg_dy == 0:
                sym = prev_sym
                if sym is None:
                    sym = _first_nonzero_direction(seg)
                if sym is None:
                    for j in range(i + 1, n_segments):
                        future_seg = seq[indices[j]:indices[j + 1]]
                        sym = _first_nonzero_direction(future_seg)
                        if sym is not None:
                            break
                if sym is None:
                    sym = 'R'
            else:
                sym = discretize_4dir(avg_dx, avg_dy)

            result_syms.append(sym)
            prev_sym = sym

        return result_syms

    for digit, samples in data.items():
        for seq in samples:
            # Original format: [(dx, dy, pen), ...]
            seq = seq[1:]

            if allow_segment:
                seg_len = _choose_segment_count(
                    seq_len=len(seq),
                    n_segments=n_segments,
                    base_segments=base_segments,
                    segment_radius=segment_radius,
                    random_segment_length=random_segment_length,
                    rng=rng,
                )
                seq = segment_average_direction(seq, n_segments=seg_len)
            else:
                seq = [discretize_4dir(dx, dy) for dx, dy, pen in seq if not (dx == 0 and dy == 0)]
                if not seq:
                    seq = ['R']

            if compress_repeats:
                seq = compress_stroke_sequence(seq)

            X.append(seq)
            y.append(digit)

    return train_test_split(X, y, test_size=0.2, random_state=random_state)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist_stroke_sequences(
        allow_segment=True,
        base_segments=20,
        segment_radius=5,
        random_segment_length=True,
        compress_repeats=False,
        random_state=42,
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Example sequence: {X_train[0]}")
    print(f"Alphabet sample: {sorted(set(X_train[0]))}")