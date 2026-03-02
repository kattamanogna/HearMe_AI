"""Training script template for text emotion + intent classifier."""

from __future__ import annotations

import argparse


def train_text_model(train_path: str, output_dir: str, epochs: int = 3) -> None:
    """Train a multitask text model.

    Next steps:
    1. Load processed data from train_path.
    2. Build tokenizer + transformer/sequence model.
    3. Optimize jointly for emotion and intent heads.
    4. Save artifacts (weights, tokenizer, labels) to output_dir.
    """
    # TODO: Implement training pipeline (PyTorch/Transformers).
    print(f"[TODO] Train text model using {train_path} for {epochs} epochs.")
    print(f"[TODO] Save model artifacts to {output_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text emotion + intent model")
    parser.add_argument("--train-path", required=True, help="Path to prepared training data")
    parser.add_argument("--output-dir", required=True, help="Directory to save model artifacts")
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_text_model(args.train_path, args.output_dir, args.epochs)
