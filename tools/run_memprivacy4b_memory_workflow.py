from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(
    "/mnt/infini-data/test/quan_space/codespace/memprivate/models/MemPrivacy-4B-RL-hf"
)
DEFAULT_BGE_PATH = Path("/mnt/infini-data/test/quan_space/codespace/memprivate/models/bge-m3")
DEFAULT_RUNTIME_CONFIG = REPO_ROOT / "evaluation/runtime_configs/eval_config.deepseek.cuda0.yaml"


@dataclass(frozen=True)
class ModeSpec:
    dataset: Path
    source_dataset: Path
    output_dir: Path
    mcq: bool
    user_limit: int | None = None
    description: str = ""


def _mode_spec(mode: str) -> ModeSpec:
    if mode == "smoke":
        return ModeSpec(
            dataset=REPO_ROOT / "data/memory_eval_smoke.jsonl",
            source_dataset=REPO_ROOT / "data/memory_eval_smoke.jsonl",
            output_dir=REPO_ROOT / "evaluation/results/memprivacy4b_smoke",
            mcq=True,
            user_limit=None,
            description="tiny plumbing check",
        )
    if mode == "persona5":
        return ModeSpec(
            dataset=REPO_ROOT / "data/personamem_v2_testset.jsonl",
            source_dataset=REPO_ROOT / "data/personamem_v2_testset.jsonl",
            output_dir=REPO_ROOT / "evaluation/results/memprivacy4b_persona5",
            mcq=True,
            user_limit=5,
            description="5-user PersonaMem-v2 pilot",
        )
    if mode == "full":
        return ModeSpec(
            dataset=REPO_ROOT / "data/personamem_v2_testset.jsonl",
            source_dataset=REPO_ROOT / "data/personamem_v2_testset.jsonl",
            output_dir=REPO_ROOT / "evaluation/results/memprivacy4b_personamem_full",
            mcq=True,
            user_limit=None,
            description="full released PersonaMem-v2 run",
        )
    raise ValueError(f"Unsupported mode: {mode}")


def _quote(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _run(command: Sequence[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("$", _quote(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _write_limited_dataset(source: Path, destination: Path, user_limit: int | None, dry_run: bool) -> Path:
    if user_limit is None:
        return source
    print(f"# materialize first {user_limit} users: {destination}", flush=True)
    if dry_run:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for index, line in enumerate(src):
            if index >= user_limit:
                break
            if line.strip():
                dst.write(line)
    return destination


def _require_path(path: Path, label: str, dry_run: bool) -> None:
    if dry_run:
        return
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _build_env(gpu: str, runtime_config: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    if runtime_config is not None:
        env["MEMPRIVACY_EVAL_CONFIG"] = str(runtime_config.resolve())
    return env


def _materialize_runtime_config_command(args: argparse.Namespace, runtime_config: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.materialize_eval_config",
        "--profile",
        str(args.eval_profile),
        "--output",
        str(runtime_config),
        "--embedding-model",
        str(args.embedding_model),
        "--embedding-device",
        f"cuda:{args.gpu}",
        "--output-path",
        str(args.memory_output_path),
        "--print-export",
    ]


def _append_memory_system_commands(
    commands: list[list[str]],
    benchmark_path: Path,
    args: argparse.Namespace,
    spec: ModeSpec,
) -> None:
    if not args.run_memory_systems:
        return
    common = [
        "--input",
        str(benchmark_path),
        "--no-mask",
        "--annotation-source",
        "model",
        "--num-workers",
        str(args.num_workers),
    ]
    if spec.mcq:
        common.append("--mcq")
    if spec.user_limit is not None:
        common.extend(["--user-num", str(spec.user_limit)])
    for system in args.memory_system:
        commands.append([sys.executable, "-m", f"evaluation.eval_{system}", *common])


def build_commands(args: argparse.Namespace) -> tuple[ModeSpec, list[list[str]]]:
    spec = _mode_spec(args.mode)
    output_dir = args.output_dir or spec.output_dir
    runtime_config = args.runtime_config
    extractor_input = output_dir / f"{args.mode}_input.jsonl"
    predictions = output_dir / f"{args.mode}_memprivacy4b_predictions.jsonl"
    extractor_metrics = output_dir / f"{args.mode}_memprivacy4b_extractor_metrics.json"
    public_records = output_dir / f"{args.mode}_public_records.jsonl"
    public_metrics = output_dir / f"{args.mode}_public_metrics.json"
    public_state = output_dir / f"{args.mode}_public_state"
    public_benchmark = output_dir / f"{args.mode}_public_benchmark.jsonl"
    audit_report = output_dir / f"{args.mode}_adversarial_audit.json"

    commands: list[list[str]] = []
    if args.materialize_config:
        commands.append(_materialize_runtime_config_command(args, runtime_config))

    commands.append(
        [
            sys.executable,
            "-m",
            "evaluation.eval",
            "--input",
            str(extractor_input if spec.user_limit is not None else spec.dataset),
            "--output",
            str(predictions),
            "--metrics-output",
            str(extractor_metrics),
            "--backend",
            "vllm",
            "--model",
            str(args.model_path),
        ]
    )
    commands.append(
        [
            sys.executable,
            "-m",
            "evaluation.eval_public_memory",
            "--input",
            str(predictions),
            "--output",
            str(public_records),
            "--metrics-output",
            str(public_metrics),
            "--state-dir",
            str(public_state),
            "--cloud-safe-dataset-output",
            str(public_benchmark),
            "--annotation-source",
            "model",
            "--minimum-token-reduction",
            str(args.minimum_token_reduction),
        ]
    )
    audit_command = [
        sys.executable,
        "-m",
        "tools.adversarial_audit",
        "--source",
        str(spec.source_dataset),
    ]
    if spec.user_limit is not None:
        audit_command.extend(["--source-user-limit", str(spec.user_limit)])
    audit_command.extend(
        [
            "--artifact",
            str(public_records),
            "--artifact",
            str(public_benchmark),
            "--report",
            str(audit_report),
        ]
    )
    commands.append(audit_command)
    _append_memory_system_commands(commands, public_benchmark, args, spec)
    return spec, commands


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the default MemPrivacy-4B-RL -> public-memory workflow on GPU 0."
    )
    parser.add_argument("--mode", choices=("smoke", "persona5", "full"), default="persona5")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value; default: 0")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_BGE_PATH)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--eval-profile", type=Path, default=REPO_ROOT / "evaluation/eval_config.deepseek.yaml")
    parser.add_argument("--memory-output-path", type=Path, default=REPO_ROOT / "evaluation/results")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--minimum-token-reduction", type=float, default=-1.0)
    parser.add_argument("--materialize-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-memory-systems", action="store_true")
    parser.add_argument(
        "--memory-system",
        action="append",
        choices=("mem0", "langmem", "memobase"),
        default=[],
        help="memory system to run after public-memory compilation; repeatable",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.model_path = args.model_path.expanduser().resolve()
    args.embedding_model = args.embedding_model.expanduser().resolve()
    args.runtime_config = args.runtime_config.expanduser()
    if not args.runtime_config.is_absolute():
        args.runtime_config = (REPO_ROOT / args.runtime_config).resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.eval_profile = args.eval_profile.expanduser().resolve()
    args.memory_output_path = args.memory_output_path.expanduser().resolve()
    if args.run_memory_systems and not args.memory_system:
        args.memory_system = ["mem0", "langmem"]

    spec, commands = build_commands(args)
    output_dir = args.output_dir or spec.output_dir
    env = _build_env(args.gpu, args.runtime_config if args.materialize_config else None)

    print(f"# mode: {args.mode} ({spec.description})")
    print(f"# gpu: {args.gpu}")
    print(f"# model: {args.model_path}")
    print(f"# output_dir: {output_dir}")
    _require_path(args.model_path, "MemPrivacy-4B-RL model", args.dry_run)
    _require_path(args.embedding_model, "BGE embedding model", args.dry_run)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    _write_limited_dataset(spec.dataset, output_dir / f"{args.mode}_input.jsonl", spec.user_limit, args.dry_run)
    for command in commands:
        _run(command, env=env, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
