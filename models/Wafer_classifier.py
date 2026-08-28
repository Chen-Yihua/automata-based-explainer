"""
Train a Wafer symbolic-sequence classifier using Wafer_loader.py.
Supports variable compressed lengths controlled by:
    base_segments ± segment_radius
"""

import sys
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, 'external_modules')
EXPLAINING_FA = os.path.join(EXTERNAL_MODULES, 'Explaining-FA')

for path in [SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset.Wafer_loader import load_Wafer_sequences
from models.sequence_classifier import SequenceClassifier


def to_object_array(X):
    arr = np.empty(len(X), dtype=object)
    for i, seq in enumerate(X):
        arr[i] = seq
    return arr


def main():
    print("=" * 60)
    print("Training Wafer Symbolic Sequence Classifier")
    print("=" * 60)

    # --- Data config ---
    alphabet_size = 7
    discretize_method = "quantile"     # try "sax" too
    use_paa = True

    # New variable-length compression setting:
    base_segments = 15
    segment_radius = 5
    random_segment_length = True

    # Optional fallback if random_segment_length=False
    n_segments = None

    compress = False
    normalize_per_sequence = True
    random_state = 42

    # Usually keep pad_to_length=None so sequences remain variable-length.
    # If your SequenceClassifier later requires fixed-length input, set this to:
    # base_segments + segment_radius
    pad_to_length = None

    print("\nLoading Wafer symbolic sequences...")
    print(f"Compressed length setting: [{base_segments - segment_radius}, {base_segments + segment_radius}]")
    X_train, X_test, y_train, y_test = load_Wafer_sequences(
        data_dir=os.path.join(PROJECT_ROOT, 'datasets', 'Wafer'),
        alphabet_size=alphabet_size,
        discretize_method=discretize_method,
        use_paa=use_paa,
        n_segments=n_segments,
        base_segments=base_segments,
        segment_radius=segment_radius,
        random_segment_length=random_segment_length,
        compress=compress,
        normalize_per_sequence=normalize_per_sequence,
        pad_to_length=pad_to_length,
        random_state=random_state,
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
    print(f"Train positive ratio: {np.mean(y_train):.4f}")
    print(f"Test positive ratio: {np.mean(y_test):.4f}")
    print(f"Example sequence: {X_train[0]}")

    max_sequence_length = (base_segments + segment_radius) if random_segment_length else max(len(seq) for seq in X_train)

    print("\nTraining classifier...")
    clf = SequenceClassifier(
        model_type='rnn',
        max_len=max_sequence_length,
        embedding_dim=8,
        rnn_units=32,
        num_layers=1,
        dropout=0.5,
        device='cuda'
    )

    clf.fit(X_train, y_train, epochs=12, batch_size=128)

    print("\nEvaluating on train/test...")
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred_test, digits=4))

    model_save_path = os.path.join(PROJECT_ROOT, 'models', 'wafer_classifier_trained.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    clf.save(model_save_path)

    split_save_path = os.path.join(PROJECT_ROOT, 'models', 'wafer_train_test_split.pkl')
    with open(split_save_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'alphabet_size': alphabet_size,
            'discretize_method': discretize_method,
            'use_paa': use_paa,
            'n_segments': n_segments,
            'base_segments': base_segments,
            'segment_radius': segment_radius,
            'random_segment_length': random_segment_length,
            'compress': compress,
            'pad_to_length': pad_to_length,
            'normalize_per_sequence': normalize_per_sequence,
            'random_state': random_state,
            'max_sequence_length': max_sequence_length,
        }, f)

    # print 5 examples sequences for each labels
    print("\nExample sequences and labels:")
    unique_labels = sorted(set(y_train))
    for label in unique_labels:
        print(f"\nLabel {label}:")
        examples = [seq for seq, lbl in zip(X_train, y_train) if lbl == label][:5]
        for i, seq in enumerate(examples):
            print(f"  Example {i+1}: {seq}")

    print(f"Train/test split saved to: {split_save_path}")
    print("\n" + "=" * 60)
    print(f"Model saved to: {model_save_path}")
    print("=" * 60)

    return clf, model_save_path


if __name__ == '__main__':
    main()
