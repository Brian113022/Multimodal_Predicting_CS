"""Train, evaluate and export multimodal subsidence models.

Run ``python train/train.py --help`` for data and output path options.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Sequence


# Make package imports work for both direct execution and ``python -m train.train``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.settings import TrainingConfig


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        help="Feature-table folder, or a data folder containing processed/. Defaults to the repository data/processed/.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Experiment output folder. Smoke tests default to a new outputs/smoke_tests/ folder.",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path,
        help="Model export folder. Defaults to artifacts/; smoke tests use <output-dir>/artifacts/.",
    )
    parser.add_argument(
        "--n-trials", type=int,
        help="Optuna trials per shortlisted model (default: 50, or 1 with --smoke-test).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Exercise all models and five-fold CV with reduced tuning, bootstrap and SHAP. Verification results only.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = TrainingConfig.create(**vars(args))
    warnings.filterwarnings("ignore")
    # Parse paths before importing the ML stack, so --help is fast.
    from train.workflow import run_training

    run_training(config)


if __name__ == "__main__":
    main()
