from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


def _resolve_imports():
    if __package__:
        from .lora import DEFAULT_LORA_TARGETS
        from .engine import train_model

        return DEFAULT_LORA_TARGETS, train_model

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from microseg.train.lora import DEFAULT_LORA_TARGETS
    from microseg.train.engine import train_model

    return DEFAULT_LORA_TARGETS, train_model


def build_parser(default_lora_targets: Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified SAM LoRA training")
    parser.add_argument("--image-dir", type=Path, required=False)
    parser.add_argument("--mask-dir", type=Path, required=False)
    parser.add_argument("--backend", choices=["meta", "hf"], default="meta", help="SAM backend to use")
    parser.add_argument("--sam-checkpoint", type=Path, default=None, help="Meta SAM checkpoint (.pth)")
    parser.add_argument("--sam-model-type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--hf-model-id", type=str, default="facebook/sam-vit-base", help="HuggingFace SAM model id")
    parser.add_argument(
        "--hf-input-size",
        type=int,
        default=1024,
        help="Resize image/mask to NxN for hf backend (0 disables explicit resize)",
    )
    parser.add_argument("--output-dir", type=Path, required=False, help="Root directory for run outputs")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard directory (relative to run dir unless absolute)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size when backend=hf")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-targets",
        nargs="+",
        default=list(default_lora_targets),
        help="Linear module suffixes to wrap with LoRA adapters",
    )
    parser.add_argument("--negative-prob", type=float, default=0.3)
    parser.add_argument("--ring-width", type=int, default=9)
    parser.add_argument("--instances-per-image", type=int, default=10)
    parser.add_argument(
        "--sample-mode",
        choices=["instance", "instance_all", "image"],
        default="instance",
        help="Use per-instance sampling, exhaustive instance enumeration, or full-image masks",
    )
    parser.add_argument(
        "--freeze-config",
        choices=["vit_prompt", "prompt_mask", "none", "custom"],
        default="vit_prompt",
        help="Freeze preset (custom uses --freeze-* flags)",
    )
    parser.add_argument("--freeze-image-encoder", action="store_true", help="Freeze image encoder (custom)")
    parser.add_argument("--freeze-prompt-encoder", action="store_true", help="Freeze prompt encoder (custom)")
    parser.add_argument("--freeze-mask-decoder", action="store_true", help="Freeze mask decoder (custom)")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--k-fold", type=int, default=1, help="Use k-fold cross validation (k>=2)")
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Run a single fold index (1-based) when k-fold is enabled",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-best", action="store_true")

    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--hf-mask-threshold", type=float, default=None, help="Deprecated alias for --mask-threshold")
    parser.add_argument("--sweep-config", type=Path, default=None, help="JSON sweep config for multiple runs")
    return parser


def _apply_overrides(args: argparse.Namespace, overrides: dict) -> argparse.Namespace:
    if not overrides:
        return args
    path_fields = {"image_dir", "mask_dir", "output_dir", "sam_checkpoint", "log_dir"}
    args_dict = vars(args)
    for key, value in overrides.items():
        if key not in args_dict:
            print(f"[WARN] Unknown sweep key: {key} (ignored)")
            continue
        if key in path_fields and value is not None:
            value = Path(value)
        args_dict[key] = value
    return argparse.Namespace(**args_dict)


def _validate_required(args: argparse.Namespace) -> None:
    missing = []
    if args.image_dir is None:
        missing.append("--image-dir")
    if args.mask_dir is None:
        missing.append("--mask-dir")
    if args.output_dir is None:
        missing.append("--output-dir")
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")


def _load_sweep_config(path: Path) -> tuple[dict, list]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {}, data
    if not isinstance(data, dict):
        raise ValueError("Sweep config must be a dict or list")
    base = data.get("base", {})
    runs = data.get("runs", [])
    if not isinstance(base, dict) or not isinstance(runs, list):
        raise ValueError("Sweep config must contain dict 'base' and list 'runs'")
    return base, runs


def main(argv: Optional[Sequence[str]] = None) -> None:
    default_targets, train_model = _resolve_imports()
    parser = build_parser(default_targets)
    args = parser.parse_args(argv)
    if args.sweep_config:
        base_overrides, runs = _load_sweep_config(args.sweep_config)
        base_args = _apply_overrides(args, base_overrides)
        if not runs:
            raise ValueError("Sweep config has no runs")
        for idx, overrides in enumerate(runs, start=1):
            if not isinstance(overrides, dict):
                raise ValueError("Each run in sweep config must be a dict")
            run_args = _apply_overrides(base_args, overrides)
            _validate_required(run_args)
            print(f"[INFO] Sweep run {idx}/{len(runs)}: {list(overrides.keys())}")
            train_model(run_args)
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return

    _validate_required(args)
    train_model(args)


if __name__ == "__main__":
    main()
