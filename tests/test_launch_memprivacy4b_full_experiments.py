from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.launch_memprivacy4b_full_experiments import (
    DATASET_SPECS,
    DatasetSpec,
    _build_extract_jobs,
    _memory_system_commands,
    _parse_csv,
    concatenate_jsonl,
    parse_args,
    split_jsonl_by_line,
    write_combined_extractor_metrics,
)


def test_parse_csv_and_default_gpus():
    args = parse_args(["--dry-run"])

    assert args.gpus == "0,1,2,3"
    assert _parse_csv("0, 1,,2") == ["0", "1", "2"]


def test_split_jsonl_by_line_round_robin(tmp_path: Path):
    source = tmp_path / "input.jsonl"
    source.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

    shards = split_jsonl_by_line(source, tmp_path / "shards", 2, dry_run=False)

    assert [path.read_text(encoding="utf-8") for path in shards] == ["a\nc\ne\n", "b\nd\n"]


def test_concatenate_jsonl_skips_blank_lines(tmp_path: Path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    first.write_text('{"a": 1}\n\n', encoding="utf-8")
    second.write_text('{"b": 2}\n', encoding="utf-8")

    concatenate_jsonl([first, second], output)

    assert output.read_text(encoding="utf-8") == '{"a": 1}\n{"b": 2}\n'


def test_write_combined_extractor_metrics(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "combined.json"
    first.write_text(json.dumps({"false_prediction_count": 2}), encoding="utf-8")
    second.write_text(json.dumps({"false_prediction_count": 3}), encoding="utf-8")

    write_combined_extractor_metrics([first, second], output)

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["false_prediction_count"] == 5
    assert result["shard_count"] == 2


def test_build_extract_jobs_uses_one_shard_per_gpu(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n{}\n{}\n{}\n", encoding="utf-8")
    spec = DatasetSpec("demo", dataset, mcq=True, description="demo")
    args = argparse.Namespace(dry_run=False)

    jobs = _build_extract_jobs(args, spec, tmp_path / "out", ["0", "1", "2", "3"])

    assert [job.gpu for job in jobs] == ["0", "1", "2", "3"]
    assert len(jobs) == 4
    assert all(job.input_path.exists() for job in jobs)


def test_memory_system_commands_add_mcq_for_personamem(tmp_path: Path):
    args = argparse.Namespace(
        run_memory_systems=True,
        memory_system=["mem0"],
        num_workers=2,
    )

    command = _memory_system_commands(args, DATASET_SPECS["personamem_v2"], tmp_path)[0]

    assert "evaluation.eval_mem0" in command
    assert "--mcq" in command
    assert "--annotation-source" in command
    assert "model" in command
