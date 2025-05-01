"""Main entry point for CNN + BiGRU + SVM Emotion Classification (TF 1.x compatibility)"""
from __future__ import absolute_import, division, print_function

__version__ = "0.1.0"
__author__ = "OUAKIB Amine"

import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from models.gru_svm.cnn_gru_svm import GruSvm
import warnings
warnings.filterwarnings("ignore")


# Hyperparameters (tune as needed)
BATCH_SIZE = 128
CELL_SIZE = 64
DROPOUT_P_KEEP = 0.3
HM_EPOCHS = 1000
LEARNING_RATE = 1e-4
N_CLASSES = 6            # sadness, joy, love, anger, fear, surprise
SEQUENCE_LENGTH = 50     # must match how you padded your data
SVM_C = 1.5
VOCAB_SIZE = 10000       # same as used in preprocessing
EMBEDDING_DIM = 100      # same as used in preprocessing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or test CNN+BiGRU+SVM for Emotion Classification"
    )
    parser.add_argument(
        "-o", "--operation",
        required=True,
        choices=["train", "test"],
        help="Operation to perform: 'train' or 'test'"
    )
    parser.add_argument(
        "-t", "--train_dataset",
        type=str,
        help="Path to training features .npy (X_train.npy)"
    )
    parser.add_argument(
        "--train_labels",
        type=str,
        help="Path to training labels .npy (y_train.npy)"
    )
    parser.add_argument(
        "-v", "--validation_dataset",
        required=True,
        type=str,
        help="Path to validation/test features .npy (X_val.npy)"
    )
    parser.add_argument(
        "--validation_labels",
        required=True,
        type=str,
        help="Path to validation/test labels .npy (y_val.npy)"
    )
    parser.add_argument(
        "-c", "--checkpoint_path",
        required=True,
        type=str,
        help="Directory to save/load model checkpoints"
    )
    parser.add_argument(
        "-l", "--log_path",
        type=str,
        help="Directory to save TensorBoard logs (training only)"
    )
    parser.add_argument(
        "-m", "--model_name",
        type=str,
        default="cnn_bigru_svm",
        help="Filename prefix for saved model checkpoints"
    )
    parser.add_argument(
        "-r", "--result_path",
        required=True,
        type=str,
        help="Directory to save prediction results"
    )
    return parser.parse_args()


def prepare_data(X_path, y_path, batch_size):
    """Load features and labels from .npy, truncate to multiple of batch_size."""
    X = np.load(X_path)
    y = np.load(y_path)
    assert X.shape[0] == y.shape[0], "Features and labels must have same number of samples"
    num = X.shape[0] - (X.shape[0] % batch_size)
    return X[:num], y[:num], num


def main(args):
    # instantiate model
    model = GruSvm(
        alpha=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        cell_size=CELL_SIZE,
        dropout_rate=DROPOUT_P_KEEP,
        num_classes=N_CLASSES,
        sequence_length=SEQUENCE_LENGTH,
        svm_c=SVM_C,
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM
    )

    if args.operation == "train":
        # require train set and log_path
        if not args.train_dataset or not args.train_labels or not args.log_path:
            raise ValueError("Training requires --train_dataset, --train_labels, and --log_path")

        # prepare training data
        X_train, y_train, train_size = prepare_data(
            args.train_dataset, args.train_labels, BATCH_SIZE
        )
        # prepare validation data
        X_val, y_val, val_size = prepare_data(
            args.validation_dataset, args.validation_labels, BATCH_SIZE
        )

        print(f"Training samples:   {train_size}")
        print(f"Validation samples: {val_size}")

        model.train(
            checkpoint_path=args.checkpoint_path,
            log_path=args.log_path,
            model_name=args.model_name,
            epochs=HM_EPOCHS,
            train_data=[X_train, y_train],
            train_size=train_size,
            validation_data=[X_val, y_val],
            validation_size=val_size,
            result_path=args.result_path
        )

    else:  # test
        X_test, y_test, test_size = prepare_data(
            args.validation_dataset, args.validation_labels, BATCH_SIZE
        )
        print(f"Testing samples: {test_size}")
        GruSvm.predict(
            batch_size=BATCH_SIZE,
            cell_size=CELL_SIZE,
            dropout_rate=DROPOUT_P_KEEP,
            num_classes=N_CLASSES,
            test_data=[X_test, y_test],
            test_size=test_size,
            checkpoint_path=args.checkpoint_path,
            result_path=args.result_path
        )


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    if args.log_path:
        os.makedirs(args.log_path, exist_ok=True)
    main(args)
