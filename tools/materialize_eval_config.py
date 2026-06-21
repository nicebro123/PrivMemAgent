from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

_SECRET_KEY_SUFFIXES = ("api_key", "apikey", "secret", "token", "password")


def _parse_scalar(value: str) -> Any:
    """Parse CLI override values with YAML scalars/lists/dicts when possible."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must use dotted.key=value syntax: {raw!r}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override key cannot be empty: {raw!r}")
    return key, _parse_scalar(value)


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor: dict[str, Any] = config
    parts = dotted_key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Invalid dotted key: {dotted_key!r}")
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        if not isinstance(current, dict):
            raise ValueError(
                f"Cannot set {dotted_key!r}: {part!r} already contains a non-object value"
            )
        cursor = current
    cursor[parts[-1]] = value


def _contains_literal_secret(dotted_key: str, value: Any) -> bool:
    key = dotted_key.lower().replace("-", "_")
    if not key.endswith(_SECRET_KEY_SUFFIXES):
        return False
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return bool(stripped) and not stripped.startswith("$")


def _absolute_cli_path(path: str, base_dir: Path | None = None) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir or Path.cwd()) / candidate
    return str(candidate.resolve())


def materialize_config(
    profile_path: Path,
    *,
    output_path: str | None = None,
    embedding_device: str | None = None,
    embedding_model: str | None = None,
    overrides: Iterable[str] = (),
    allow_literal_secret: bool = False,
) -> dict[str, Any]:
    profile = profile_path.expanduser().resolve()
    with profile.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Profile must contain a YAML mapping: {profile}")
    config = copy.deepcopy(config)

    if embedding_device:
        config.setdefault("embedding_model", {})["device"] = embedding_device
    if embedding_model:
        config.setdefault("embedding_model", {})["model"] = _absolute_cli_path(embedding_model)
    if output_path:
        # Memory runners resolve output_path relative to the config file directory.
        # Runtime configs therefore store an absolute result root to make generated
        # configs relocatable and avoid accidentally writing under runtime_configs/.
        config["output_path"] = _absolute_cli_path(output_path)

    for raw_override in overrides:
        dotted_key, value = _parse_override(raw_override)
        if _contains_literal_secret(dotted_key, value) and not allow_literal_secret:
            raise ValueError(
                f"Refusing to write a literal secret for {dotted_key!r}. "
                "Use an environment reference such as $OPENAI_API_KEY, or pass "
                "--allow-literal-secret only for trusted local debugging."
            )
        _set_dotted(config, dotted_key, value)

    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a runtime evaluation YAML from a tracked profile. "
            "Use this for per-server GPU/model/result-path settings without "
            "committing secrets or generated configs."
        )
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("evaluation/eval_config.yaml"),
        help="Tracked base profile, e.g. evaluation/eval_config.deepseek.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Runtime config path to write. Keep this under an ignored directory.",
    )
    parser.add_argument("--embedding-device", help="Override embedding_model.device, e.g. cuda:0")
    parser.add_argument("--embedding-model", help="Override embedding_model.model checkpoint path")
    parser.add_argument(
        "--output-path",
        help="Result root for memory-system runners. Relative values are stored as absolute paths.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="DOTTED.KEY=VALUE",
        help="Apply a YAML-parsed dotted-key override. Repeatable.",
    )
    parser.add_argument(
        "--allow-literal-secret",
        action="store_true",
        help="Allow literal api_key/token/password overrides. Prefer environment references.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated config instead of writing it.",
    )
    parser.add_argument(
        "--print-export",
        action="store_true",
        help="After writing, print export MEMPRIVACY_EVAL_CONFIG=...",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = materialize_config(
        args.profile,
        output_path=args.output_path,
        embedding_device=args.embedding_device,
        embedding_model=args.embedding_model,
        overrides=args.overrides,
        allow_literal_secret=args.allow_literal_secret,
    )
    serialized = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if args.dry_run:
        print(serialized, end="")
        return 0

    output = args.output.expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(f"Wrote runtime config: {output}")
    if args.print_export:
        print(f"export MEMPRIVACY_EVAL_CONFIG={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
