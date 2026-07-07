import os
import pickle
import sys

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

from dataset.mnist_stroke_loader import load_mnist_stroke_sequences
from models.sequence_classifier import SequenceClassifier


def to_object_array(sequences):
    arr = np.empty(len(sequences), dtype=object)
    for i, seq in enumerate(sequences):
        arr[i] = seq
    return arr


def main():
    print("=" * 60)
    print("Training MNIST 4-Direction Stroke Sequence Classifier")
    print("=" * 60)

    base_segments = 15
    segment_radius = 5
    random_segment_length = True

    min_len = max(1, base_segments - segment_radius)
    max_len = base_segments + segment_radius

    print(
        f"\nLoading MNIST stroke sequences "
        f"(target length in [{min_len}, {max_len}], "
        f"random_segment_length={random_segment_length}, alphabet=4 directions)..."
    )

    X_train, X_test, y_train, y_test = load_mnist_stroke_sequences(
        allow_segment=True,
        n_segments=base_segments,
        base_segments=base_segments,
        segment_radius=segment_radius,
        random_segment_length=random_segment_length,
        compress_repeats=False,
        random_state=42,
    )

    X_train = to_object_array(X_train)
    X_test = to_object_array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    train_lengths = [len(seq) for seq in X_train]
    test_lengths = [len(seq) for seq in X_test]
    print("\nSequence length distribution:")
    print(f"  Train -> min: {min(train_lengths)}, max: {max(train_lengths)}, mean: {np.mean(train_lengths):.2f}")
    print(f"  Test  -> min: {min(test_lengths)}, max: {max(test_lengths)}, mean: {np.mean(test_lengths):.2f}")
    print(f"  Example sequence: {X_train[0]}")

    max_sequence_length = max_len

    print("\nTraining classifier...")
    clf = SequenceClassifier(
        model_type='rnn',
        max_len=max_sequence_length,
        embedding_dim=64,
        rnn_units=128,
        num_layers=2,
        dropout=0.5,
        device='cuda'
    )

    clf.fit(X_train, y_train, epochs=50, batch_size=128)

    print("\nEvaluating on test set...")
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    model_save_path = os.path.join(PROJECT_ROOT, "models", "mnist_classifier_trained.pth")
    clf.save(model_save_path)

    split_save_path = os.path.join(PROJECT_ROOT, "models", "mnist_train_test_split.pkl")
    with open(split_save_path, "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "base_segments": base_segments,
            "segment_radius": segment_radius,
            "random_segment_length": random_segment_length,
            "alphabet": ['R', 'U', 'L', 'D'],
        }, f)

    # print 5 random examples for each class
    print("\nSample sequences from training set:")
    for label in sorted(set(y_train)):
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


if __name__ == "__main__":
    main()