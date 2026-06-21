from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(
    "/mnt/infini-data/test/quan_space/codespace/memprivate/models/MemPrivacy-4B-RL-hf"
)
DEFAULT_BGE_PATH = Path("/mnt/infini-data/test/quan_space/codespace/memprivate/models/bge-m3")
DEFAULT_EVAL_PROFILE = REPO_ROOT / "evaluation/eval_config.deepseek.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evaluation/results/memprivacy4b_full"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset: Path
    mcq: bool
    description: str


@dataclass(frozen=True)
class ExtractJob:
    dataset: str
    shard_index: int
    gpu: str
    input_path: Path
    prediction_path: Path
    metrics_path: Path
    log_path: Path


DATASET_SPECS: dict[str, DatasetSpec] = {
    "personamem_v2": DatasetSpec(
        name="personamem_v2",
        dataset=REPO_ROOT / "data/personamem_v2_testset.jsonl",
        mcq=True,
        description="full released PersonaMem-v2",
    ),
    "memprivacy_bench": DatasetSpec(
        name="memprivacy_bench",
        dataset=REPO_ROOT / "data/memprivacy_bench_testset.jsonl",
        mcq=False,
        description="full released MemPrivacy-Bench",
    ),
}


def _quote(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _require_path(path: Path, label: str, dry_run: bool) -> None:
    if dry_run:
        return
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _read_jsonl(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as source:
        return [line for line in source if line.strip()]


def split_jsonl_by_line(source: Path, shard_dir: Path, shard_count: int, dry_run: bool) -> list[Path]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    lines = _read_jsonl(source) if source.exists() else []
    shard_paths = [shard_dir / f"shard_{index:02d}.jsonl" for index in range(shard_count)]
    if dry_run:
        return shard_paths
    shard_dir.mkdir(parents=True, exist_ok=True)
    handles = [path.open("w", encoding="utf-8") for path in shard_paths]
    try:
        for line_index, line in enumerate(lines):
            handles[line_index % shard_count].write(line)
    finally:
        for handle in handles:
            handle.close()
    return shard_paths


def concatenate_jsonl(inputs: Sequence[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for path in inputs:
            if not path.exists():
                raise FileNotFoundError(f"Missing shard output: {path}")
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        destination.write(line)


def write_combined_extractor_metrics(inputs: Sequence[Path], output: Path) -> None:
    metrics = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing shard metrics: {path}")
        metrics.append(json.loads(path.read_text(encoding="utf-8")))
    total_failures = sum(int(metric.get("false_prediction_count", 0)) for metric in metrics)
    result = {
        "false_prediction_count": total_failures,
        "shard_count": len(metrics),
        "shard_metrics": [str(path.resolve()) for path in inputs],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _runtime_config_command(args: argparse.Namespace, runtime_config: Path, gpu: str) -> list[str]:
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
        f"cuda:{gpu}",
        "--output-path",
        str(args.output_root),
        "--print-export",
    ]


def _extract_command(args: argparse.Namespace, job: ExtractJob) -> list[str]:
    return [
        sys.executable,
        "-m",
        "evaluation.eval",
        "--input",
        str(job.input_path),
        "--output",
        str(job.prediction_path),
        "--metrics-output",
        str(job.metrics_path),
        "--backend",
        "vllm",
        "--model",
        str(args.model_path),
    ]


def _public_memory_command(args: argparse.Namespace, dataset_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "evaluation.eval_public_memory",
        "--input",
        str(dataset_dir / "predictions.jsonl"),
        "--output",
        str(dataset_dir / "public_records.jsonl"),
        "--metrics-output",
        str(dataset_dir / "public_metrics.json"),
        "--state-dir",
        str(dataset_dir / "public_state"),
        "--cloud-safe-dataset-output",
        str(dataset_dir / "public_benchmark.jsonl"),
        "--annotation-source",
        "model",
        "--minimum-token-reduction",
        str(args.minimum_token_reduction),
    ]


def _audit_command(spec: DatasetSpec, dataset_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.adversarial_audit",
        "--source",
        str(spec.dataset),
        "--artifact",
        str(dataset_dir / "public_records.jsonl"),
        "--artifact",
        str(dataset_dir / "public_benchmark.jsonl"),
        "--report",
        str(dataset_dir / "adversarial_audit.json"),
    ]


def _memory_system_commands(
    args: argparse.Namespace, spec: DatasetSpec, dataset_dir: Path
) -> list[list[str]]:
    if not args.run_memory_systems:
        return []
    systems = args.memory_system or ["mem0", "langmem"]
    common = [
        "--input",
        str(dataset_dir / "public_benchmark.jsonl"),
        "--no-mask",
        "--annotation-source",
        "model",
        "--num-workers",
        str(args.num_workers),
    ]
    if spec.mcq:
        common.append("--mcq")
    return [[sys.executable, "-m", f"evaluation.eval_{system}", *common] for system in systems]


def _run_logged(
    command: Sequence[str],
    log_path: Path,
    *,
    env: dict[str, str],
    dry_run: bool,
    check: bool = True,
) -> int:
    print("$", _quote(command), flush=True)
    print(f"# log: {log_path}", flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + _quote(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=check,
        )
    return completed.returncode


def _start_logged(
    command: Sequence[str], log_path: Path, *, env: dict[str, str], dry_run: bool
) -> subprocess.Popen | None:
    print("$", _quote(command), flush=True)
    print(f"# log: {log_path}", flush=True)
    if dry_run:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    log.write("$ " + _quote(command) + "\n")
    log.flush()
    try:
        return subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    except Exception:
        log.close()
        raise


def _check_gpus_available(gpus: Sequence[str], max_memory_mb: int, dry_run: bool) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    print("$", _quote(command), flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    seen: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            seen[parts[0]] = int(parts[1])
    missing = [gpu for gpu in gpus if gpu not in seen]
    if missing:
        raise RuntimeError(f"Requested GPUs not visible: {missing}")
    busy = {gpu: seen[gpu] for gpu in gpus if seen[gpu] > max_memory_mb}
    if busy:
        raise RuntimeError(
            "Requested GPUs appear busy: "
            + ", ".join(f"gpu {gpu}: {memory} MiB" for gpu, memory in busy.items())
        )


def _write_manifest(path: Path, manifest: dict, dry_run: bool) -> None:
    print(f"# manifest: {path}", flush=True)
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _dataset_finished(dataset_dir: Path) -> bool:
    expected = [
        dataset_dir / "predictions.jsonl",
        dataset_dir / "public_records.jsonl",
        dataset_dir / "public_metrics.json",
        dataset_dir / "public_benchmark.jsonl",
        dataset_dir / "adversarial_audit.json",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in expected)


def _build_extract_jobs(
    args: argparse.Namespace, spec: DatasetSpec, dataset_dir: Path, gpus: Sequence[str]
) -> list[ExtractJob]:
    shard_dir = dataset_dir / "shards"
    prediction_dir = dataset_dir / "extractor_shards"
    metrics_dir = dataset_dir / "extractor_metrics_shards"
    log_dir = dataset_dir / "logs"
    shard_paths = split_jsonl_by_line(spec.dataset, shard_dir, len(gpus), args.dry_run)
    return [
        ExtractJob(
            dataset=spec.name,
            shard_index=index,
            gpu=gpu,
            input_path=shard_paths[index],
            prediction_path=prediction_dir / f"shard_{index:02d}_predictions.jsonl",
            metrics_path=metrics_dir / f"shard_{index:02d}_metrics.json",
            log_path=log_dir / f"extract_shard_{index:02d}_gpu{gpu}.log",
        )
        for index, gpu in enumerate(gpus)
    ]


def _run_dataset(args: argparse.Namespace, spec: DatasetSpec, gpus: Sequence[str]) -> None:
    dataset_dir = args.output_root / spec.name
    log_dir = dataset_dir / "logs"
    runtime_config = dataset_dir / "runtime_config.yaml"
    manifest_path = dataset_dir / "manifest.json"
    print(f"# dataset: {spec.name} ({spec.description})", flush=True)
    print(f"# output_dir: {dataset_dir}", flush=True)
    if args.skip_existing and _dataset_finished(dataset_dir):
        print(f"# skip existing completed dataset: {spec.name}", flush=True)
        return

    _require_path(spec.dataset, f"{spec.name} dataset", args.dry_run)
    jobs = _build_extract_jobs(args, spec, dataset_dir, gpus)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": spec.name,
        "dataset_path": str(spec.dataset.resolve()),
        "model_path": str(args.model_path),
        "embedding_model": str(args.embedding_model),
        "gpus": list(gpus),
        "annotation_source": "model",
        "jobs": [asdict(job) for job in jobs],
    }
    _write_manifest(manifest_path, manifest, args.dry_run)

    runtime_env = os.environ.copy()
    if args.materialize_config:
        runtime_env["CUDA_VISIBLE_DEVICES"] = gpus[0]
        _run_logged(
            _runtime_config_command(args, runtime_config, gpus[0]),
            log_dir / "materialize_runtime_config.log",
            env=runtime_env,
            dry_run=args.dry_run,
        )
        runtime_env["MEMPRIVACY_EVAL_CONFIG"] = str(runtime_config.resolve())

    processes: list[tuple[ExtractJob, subprocess.Popen | None]] = []
    for job in jobs:
        env = runtime_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = job.gpu
        process = _start_logged(_extract_command(args, job), job.log_path, env=env, dry_run=args.dry_run)
        processes.append((job, process))

    failed: list[tuple[ExtractJob, int]] = []
    for job, process in processes:
        if process is None:
            continue
        return_code = process.wait()
        if return_code != 0:
            failed.append((job, return_code))
    if failed:
        failures = ", ".join(
            f"{job.dataset}/shard{job.shard_index} on gpu {job.gpu}: {return_code}"
            for job, return_code in failed
        )
        raise RuntimeError(f"Extractor shard failures: {failures}")

    shard_predictions = [job.prediction_path for job in jobs]
    shard_metrics = [job.metrics_path for job in jobs]
    print(f"# merge predictions: {dataset_dir / 'predictions.jsonl'}", flush=True)
    if not args.dry_run:
        concatenate_jsonl(shard_predictions, dataset_dir / "predictions.jsonl")
        write_combined_extractor_metrics(shard_metrics, dataset_dir / "extractor_metrics.json")

    post_env = runtime_env.copy()
    post_env["CUDA_VISIBLE_DEVICES"] = gpus[0]
    _run_logged(
        _public_memory_command(args, dataset_dir),
        log_dir / "public_memory.log",
        env=post_env,
        dry_run=args.dry_run,
    )
    audit_return_code = _run_logged(
        _audit_command(spec, dataset_dir),
        log_dir / "adversarial_audit.log",
        env=post_env,
        dry_run=args.dry_run,
        check=not args.continue_on_audit_failure,
    )
    if audit_return_code != 0 and args.continue_on_audit_failure:
        print(
            f"# adversarial audit returned {audit_return_code} for {spec.name}; "
            "continuing because --continue-on-audit-failure is enabled",
            flush=True,
        )
    for command in _memory_system_commands(args, spec, dataset_dir):
        name = command[2].replace(".", "_")
        _run_logged(command, log_dir / f"{name}.log", env=post_env, dry_run=args.dry_run)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch full MemPrivacy-4B memory experiments across multiple single-GPU shards."
    )
    parser.add_argument("--gpus", default="0,1,2,3", help="comma-separated GPU IDs; default: 0,1,2,3")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(DATASET_SPECS),
        help="dataset to run; repeatable. Defaults to personamem_v2 and memprivacy_bench.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_BGE_PATH)
    parser.add_argument("--eval-profile", type=Path, default=DEFAULT_EVAL_PROFILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--minimum-token-reduction", type=float, default=-1.0)
    parser.add_argument("--materialize-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-gpus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-gpu-memory-mb", type=int, default=1024)
    parser.add_argument(
        "--continue-on-audit-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep launching later datasets when adversarial audit reports leakage. "
            "The audit report and non-zero return code remain in the logs."
        ),
    )
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
    gpus = _parse_csv(args.gpus)
    if not gpus:
        raise ValueError("At least one GPU must be specified")
    args.model_path = args.model_path.expanduser().resolve()
    args.embedding_model = args.embedding_model.expanduser().resolve()
    args.eval_profile = args.eval_profile.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    datasets = args.dataset or list(DATASET_SPECS)

    print(f"# gpus: {','.join(gpus)}")
    print(f"# model: {args.model_path}")
    print(f"# embedding: {args.embedding_model}")
    print(f"# output_root: {args.output_root}")
    _require_path(args.model_path, "MemPrivacy-4B-RL model", args.dry_run)
    _require_path(args.embedding_model, "BGE embedding model", args.dry_run)
    _require_path(args.eval_profile, "DeepSeek eval profile", args.dry_run)
    if args.check_gpus:
        _check_gpus_available(gpus, args.max_gpu_memory_mb, args.dry_run)
    for dataset_name in datasets:
        _run_dataset(args, DATASET_SPECS[dataset_name], gpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
