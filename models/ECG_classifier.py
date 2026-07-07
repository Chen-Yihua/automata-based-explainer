
"""
Train an ECG5000 symbolic-sequence classifier using ECG_loader.py.
"""

import sys
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, 'external_modules')
MODIFIED_MODULES = os.path.join(PROJECT_ROOT, 'modified_modules')
EXPLAINING_FA = os.path.join(EXTERNAL_MODULES, 'Explaining-FA')
INTERPRETERA_SRC = os.path.join(EXTERNAL_MODULES, 'interpretera', 'src')

for path in [MODIFIED_MODULES, SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, INTERPRETERA_SRC, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset.ECG_loader import load_EGG_sequences
from models.sequence_classifier import SequenceClassifier


def to_object_array(X):
    arr = np.empty(len(X), dtype=object)
    for i, seq in enumerate(X):
        arr[i] = seq
    return arr


def main():
    print("=" * 60)
    print("Training ECG5000 Symbolic Sequence Classifier")
    print("=" * 60)

    # 你可以改這三個參數來控制壓縮後的長度範圍
    base_segments = 15
    segment_radius = 5
    random_segment_length = True

    min_len = max(1, base_segments - segment_radius)
    max_len = base_segments + segment_radius

    # --- Data config ---
    alphabet_size = 7
    discretize_method = "quantile"   # try "sax" too
    use_paa = True
    compress = False
    pad_to_length = None
    normalize_per_sequence = True

    print(
        f"\nLoading ECG5000 symbolic sequences "
        f"(target length in [{min_len}, {max_len}], "
        f"random_segment_length={random_segment_length})..."
    )
    X_train, X_test, y_train, y_test = load_EGG_sequences(
        data_dir=os.path.join(PROJECT_ROOT, 'datasets', 'ECG5000'),
        alphabet_size=alphabet_size,
        discretize_method=discretize_method,
        use_paa=use_paa,
        n_segments=base_segments,  # random=False 時會用到
        base_segments=base_segments,
        segment_radius=segment_radius,
        random_segment_length=random_segment_length,
        compress=compress,
        normalize_per_sequence=normalize_per_sequence,
        pad_to_length=pad_to_length,
        random_state=42,
    )

    X_train = to_object_array(X_train)
    X_test = to_object_array(X_test)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    train_lengths = [len(seq) for seq in X_train]
    test_lengths = [len(seq) for seq in X_test]
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Train length distribution: min={min(train_lengths)}, max={max(train_lengths)}, mean={np.mean(train_lengths):.2f}")
    print(f"Test length distribution: min={min(test_lengths)}, max={max(test_lengths)}, mean={np.mean(test_lengths):.2f}")
    print(f"Example sequence: {X_train[0]}")

    max_sequence_length = max_len

    print("\nTraining classifier...")
    clf = SequenceClassifier(
        model_type='rnn',
        max_len=max_sequence_length,
        embedding_dim=16,
        rnn_units=64,
        num_layers=1,
        dropout=0.5,
        device='cuda'
    )

    clf.fit(X_train, y_train, epochs=15, batch_size=128)

    print("\nEvaluating on train/test...")
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    model_save_path = os.path.join(PROJECT_ROOT, 'models', 'ECG_classifier_trained.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    clf.save(model_save_path)

    split_save_path = os.path.join(PROJECT_ROOT, 'models', 'ECG_train_test_split.pkl')
    with open(split_save_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'alphabet_size': alphabet_size,
            'discretize_method': discretize_method,
            'use_paa': use_paa,
            'base_segments': base_segments,
            'segment_radius': segment_radius,
            'random_segment_length': random_segment_length,
            'compress': compress,
            'pad_to_length': pad_to_length,
            'normalize_per_sequence': normalize_per_sequence,
        }, f)

    # print 5 random examples for each class
    print("\nSample sequences from training set:")
    unique_labels = sorted(set(y_train))
    for label in unique_labels:
        print(f"\nLabel {label}:")
        label_indices = np.where(y_train == label)[0]
        sample_indices = np.random.choice(label_indices, size=min(5, len(label_indices)), replace=False)
        for idx in sample_indices:
            print(f"  Sequence: {X_train[idx]}, Length: {len(X_train[idx])}")

    print(f"Train/test split saved to: {split_save_path}")
    print("\n" + "=" * 60)
    print(f"Model saved to: {model_save_path}")
    print("=" * 60)

    return clf, model_save_path


if __name__ == '__main__':
    main()
