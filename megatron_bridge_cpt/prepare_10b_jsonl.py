#!/usr/bin/env python3
"""Combine downloaded corpus JSONL shards into one capped pretraining JSONL."""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 10B-token CPT JSONL from corpus_download outputs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files or globs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--target-tokens", type=int, default=10_000_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-files", action="store_true")
    parser.add_argument("--allow-missing-token-count", action="store_true")
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    unique = sorted({path.resolve() for path in paths})
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise SystemExit(f"Missing input file(s): {', '.join(missing)}")
    return unique


def get_token_count(record: dict[str, Any], allow_missing: bool) -> int:
    token_count = record.get("token_count")
    if isinstance(token_count, int):
        return token_count
    if allow_missing:
        return 0
    raise ValueError("record has no integer token_count; rerun with --allow-missing-token-count if intentional")


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    if args.shuffle_files:
        random.Random(args.seed).shuffle(paths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_docs = 0
    per_file = []

    with args.output.open("w", encoding="utf-8") as out:
        for path in paths:
            file_tokens = 0
            file_docs = 0
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if total_tokens >= args.target_tokens:
                        break
                    record = json.loads(line)
                    text = str(record.get("text", "")).strip()
                    if not text:
                        continue
                    try:
                        token_count = get_token_count(record, args.allow_missing_token_count)
                    except ValueError as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc
                    if token_count and total_tokens + token_count > args.target_tokens:
                        break
                    out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    total_tokens += token_count
                    file_tokens += token_count
                    total_docs += 1
                    file_docs += 1

            per_file.append({"path": str(path), "docs": file_docs, "tokens": file_tokens})
            print(f"{path}: wrote {file_docs:,} docs / {file_tokens:,} tokens; total={total_tokens:,}")
            if total_tokens >= args.target_tokens:
                break

    manifest = {
        "output": str(args.output),
        "target_tokens": args.target_tokens,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "inputs": per_file,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote {total_docs:,} docs / {total_tokens:,} tokens to {args.output}")
    if total_tokens < args.target_tokens:
        print(
            f"[WARN] Target was {args.target_tokens:,} tokens, "
            f"but only {total_tokens:,} tokens were available."
        )
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
