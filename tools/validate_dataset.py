import argparse
import json
from pathlib import Path


def validate_dataset(path: Path) -> list[dict]:
    issues = []
    with path.open(encoding="utf-8") as source:
        for user_line, line in enumerate(source, 1):
            if not line.strip():
                continue
            user = json.loads(line)
            for message_index, message in enumerate(user.get("dialogues", [])):
                content = message.get("content", "")
                for item_index, item in enumerate(message.get("privacy_info", [])):
                    original = item.get("original_text")
                    if not isinstance(original, str) or not original.strip():
                        reason = "empty_original_text"
                    elif original not in content:
                        reason = "span_not_in_message"
                    else:
                        continue
                    issues.append(
                        {
                            "dataset": str(path),
                            "user_line": user_line,
                            "uuid": user.get("uuid"),
                            "message_index": message_index,
                            "item_index": item_index,
                            "reason": reason,
                            "original_text": original,
                        }
                    )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark span annotations")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    issues = [
        issue for path in args.paths for issue in validate_dataset(path.expanduser().resolve())
    ]
    payload = {"issue_count": len(issues), "issues": issues}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
