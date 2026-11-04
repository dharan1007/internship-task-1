"""Explicit, fail-closed research workflow commands."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch  # noqa: F401  # Load Windows DLLs before scientific plotting libraries.

from .data.audit import audit
from .data.manifest import read_manifest, write_manifest
from .data.prepare import prepare_eyepacs, prepare_messidor, prepare_messidor2
from .data.split import split_eyepacs
from .evaluate import evaluate
from .qualify_scan import qualify_scan
from .report import generate_report
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="drvim")
    commands = parser.add_subparsers(dest="command", required=True)
    eye = commands.add_parser("prepare-eyepacs")
    eye.add_argument("--images", type=Path, required=True)
    eye.add_argument("--labels", type=Path, required=True)
    eye.add_argument("--exclusions", type=Path)
    eye.add_argument("--out", type=Path, required=True)
    split = commands.add_parser("split-eyepacs")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--out", type=Path, required=True)
    split.add_argument("--seed", type=int, required=True)
    original = commands.add_parser("prepare-messidor")
    original.add_argument("--images", type=Path, required=True)
    original.add_argument("--out", type=Path, required=True)
    second = commands.add_parser("prepare-messidor2")
    second.add_argument("--images", type=Path, required=True)
    second.add_argument("--labels", type=Path, required=True)
    second.add_argument("--pairs", type=Path, required=True)
    second.add_argument("--out", type=Path, required=True)
    auditing = commands.add_parser("audit")
    auditing.add_argument("--run-dir", type=Path, required=True)
    auditing.add_argument("--eyepacs", type=Path, required=True)
    auditing.add_argument("--messidor", type=Path, required=True)
    auditing.add_argument("--messidor2", type=Path,
                          help="Optional adjudicated five-grade external cohort")
    auditing.add_argument("--out", type=Path, required=True)
    qualifying = commands.add_parser("qualify-scan")
    qualifying.add_argument("--config", type=Path, required=True)
    qualifying.add_argument("--manifest", type=Path, required=True)
    qualifying.add_argument("--out", type=Path, required=True)
    qualifying.add_argument("--gradient-tokens", type=int, default=128)
    training = commands.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--manifest", type=Path, required=True)
    training.add_argument("--qualification", type=Path, required=True)
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--resume", action="store_true")
    training.add_argument("--stop-after-epoch", type=int,
                          help="Pause after this absolute epoch; retain full-run config and checkpoints")
    testing = commands.add_parser("evaluate")
    testing.add_argument("--run-dir", type=Path, required=True)
    testing.add_argument("--eyepacs", type=Path, required=True)
    testing.add_argument("--messidor", type=Path, required=True)
    testing.add_argument("--messidor2", type=Path,
                         help="Optional adjudicated five-grade external cohort")
    testing.add_argument("--audit", type=Path, required=True)
    testing.add_argument("--batch-size", type=int,
                         help="Default: the completed training run's configured microbatch")
    testing.add_argument("--workers", type=int,
                         help="Default: the completed training run's configured worker count")
    report = commands.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--eyepacs", type=Path, required=True)
    report.add_argument("--messidor", type=Path, required=True)
    report.add_argument("--messidor2", type=Path,
                        help="Optional adjudicated five-grade external cohort")
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-eyepacs":
        prepare_eyepacs(args.images, args.labels, args.out, args.exclusions)
    elif args.command == "split-eyepacs":
        write_manifest(split_eyepacs(read_manifest(args.manifest), args.seed), args.out)
    elif args.command == "prepare-messidor":
        prepare_messidor(args.images, args.out)
    elif args.command == "prepare-messidor2":
        prepare_messidor2(args.images, args.labels, args.out, args.pairs)
    elif args.command == "audit":
        audit(args.run_dir, args.eyepacs, args.messidor, args.messidor2, args.out)
    elif args.command == "qualify-scan":
        qualify_scan(args.config, args.manifest, args.out,
                     gradient_tokens=args.gradient_tokens)
    elif args.command == "train":
        train(args.config, args.manifest, args.qualification, args.run_dir,
              resume=args.resume, stop_after_epoch=args.stop_after_epoch)
    elif args.command == "evaluate":
        evaluate(args.run_dir, args.eyepacs, args.messidor, args.messidor2, args.audit,
                 batch_size=args.batch_size, workers=args.workers)
    elif args.command == "report":
        generate_report(args.run_dir, args.out, args.eyepacs,
                        args.messidor, args.messidor2)


if __name__ == "__main__":
    main()
