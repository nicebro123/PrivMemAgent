from __future__ import annotations

import argparse
import importlib
import json
import socket
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from openai import OpenAI

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.utils import _load_config

SYSTEM_PACKAGES = {
    "mem0": ("mem0", "chromadb"),
    "langmem": ("langchain_core", "langgraph", "langmem"),
    "memobase": ("memobase",),
}


def _valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _check_packages(systems: Iterable[str]) -> dict:
    results = {}
    for system in systems:
        missing = []
        for package in SYSTEM_PACKAGES[system]:
            try:
                importlib.import_module(package)
            except Exception as exc:
                missing.append(f"{package}: {type(exc).__name__}")
        results[system] = {"ok": not missing, "missing": missing}
    return results


def _check_memobase(project_url: str, timeout: float) -> dict:
    if not _valid_http_url(project_url):
        return {"ok": False, "error": "invalid project_url"}
    parsed = urlparse(project_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            pass
    except OSError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "host": parsed.hostname, "port": port}


def _check_openai(config: dict, timeout: float, probe: bool) -> dict:
    base_url = str(config.get("openai_base_url", "") or "")
    api_key = str(config.get("openai_api_key", "") or "")
    result = {
        "ok": False,
        "base_url_configured": _valid_http_url(base_url),
        "api_key_configured": bool(api_key),
        "probe_requested": probe,
    }
    if not result["base_url_configured"] or not result["api_key_configured"]:
        result["error"] = "OPENAI_BASE_URL and OPENAI_API_KEY must be configured"
        return result
    if not probe:
        result["ok"] = True
        return result
    try:
        models = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout).models.list()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["ok"] = True
    result["reported_model_count"] = len(models.data)
    return result


def run_preflight(
    config_path: Path,
    systems: Iterable[str],
    timeout: float,
    probe_openai: bool,
) -> dict:
    selected_systems = list(dict.fromkeys(systems))
    config = _load_config(str(config_path))
    package_results = _check_packages(selected_systems)
    result = {
        "python": {
            "version": ".".join(map(str, sys.version_info[:3])),
            "supported": sys.version_info >= (3, 10),
        },
        "packages": package_results,
        "openai": _check_openai(config, timeout, probe_openai),
    }
    if "memobase" in selected_systems:
        result["memobase"] = _check_memobase(
            str(config.get("memobase", {}).get("project_url", "") or ""),
            timeout,
        )
    result["ok"] = (
        result["python"]["supported"]
        and all(item["ok"] for item in package_results.values())
        and result["openai"]["ok"]
        and result.get("memobase", {"ok": True})["ok"]
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate memory-evaluation dependencies")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("evaluation/eval_config.yaml"),
    )
    parser.add_argument(
        "--system",
        action="append",
        choices=tuple(SYSTEM_PACKAGES),
        dest="systems",
        help="memory system to validate; repeat for multiple systems",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--probe-openai", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preflight(
        config_path=args.config.expanduser().resolve(),
        systems=args.systems or SYSTEM_PACKAGES.keys(),
        timeout=args.timeout,
        probe_openai=args.probe_openai,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
