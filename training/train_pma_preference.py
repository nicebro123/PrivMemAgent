from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return records


def validate_preference_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("preference dataset is empty")
    for index, record in enumerate(records):
        for key in ("prompt", "chosen", "rejected"):
            if not record.get(key):
                raise ValueError(f"preference record {index} missing {key}")
        if record["chosen"] == record["rejected"]:
            raise ValueError(f"preference record {index} has identical outputs")
        for key in ("chosen", "rejected"):
            try:
                target = json.loads(record[key])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"preference record {index} {key} is not JSON"
                ) from exc
            for target_key in (
                "public_memory",
                "private_residue",
                "abstraction_trace",
            ):
                if target_key not in target:
                    raise ValueError(
                        f"preference record {index} {key} missing {target_key}"
                    )


def write_manifest(
    output_dir: str,
    train_file: str,
    records: list[dict[str, Any]],
    model_name_or_path: str,
    *,
    status: str,
    dry_run: bool,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "pma_preference",
        "dry_run": dry_run,
        "model_name_or_path": model_name_or_path,
        "train_file": os.path.abspath(train_file),
        "num_records": len(records),
        "status": status,
    }
    path = os.path.join(output_dir, "pma_preference_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return path


def train(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:  # pragma: no cover - GPU environment-specific
        raise RuntimeError(
            "Preference training requires transformers, datasets, accelerate, "
            "peft, and trl"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        revision=args.model_revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        revision=args.model_revision,
    )
    dataset = Dataset.from_list(
        [
            {
                "prompt": record["prompt"],
                "chosen": record["chosen"],
                "rejected": record["rejected"],
            }
            for record in records
        ]
    )
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16,
        max_length=args.max_length,
        beta=args.beta,
    )
    peft_config = None
    if args.lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preference-tune PMA with DPO.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--model-revision",
        default="main",
        help="Prefer an immutable Hugging Face commit hash for reproducibility.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    records = read_jsonl(args.train_file)
    validate_preference_records(records)
    if args.dry_run:
        path = write_manifest(
            args.output_dir,
            args.train_file,
            records,
            args.model_name_or_path,
            status="validated",
            dry_run=True,
        )
        print(f"Wrote PMA preference validation manifest to {path}")
        return
    train(args, records)
    path = write_manifest(
        args.output_dir,
        args.train_file,
        records,
        args.model_name_or_path,
        status="trained",
        dry_run=False,
    )
    print(f"Saved PMA preference checkpoint and manifest under {args.output_dir}")
    print(path)


if __name__ == "__main__":
    main()
