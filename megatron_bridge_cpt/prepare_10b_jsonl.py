#!/usr/bin/env python3
"""Prepare capped Qwen3 CPT JSONL inputs from downloaded corpus shards."""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_VECTOR_LANGUAGE_BY_LANGUAGE = {
    "en": "eng",
    "en_instruct": "eng",
    "ja": "jpn",
    "fr": "fra",
    "es": "spa",
    "zh": "cmn",
    "de": "deu",
    "nl": "nld",
    "ru": "rus",
    "uk": "ukr",
    "pl": "pol",
    "cs": "ces",
    "pt": "por",
    "it": "ita",
    "ur": "urd",
    "fa": "pes",
    "ga": "gle",
    "cy": "cym",
    "ar": "arb",
    "he": "heb",
    "fi": "fin",
    "et": "est",
    "hu": "hun",
    "tr": "tur",
    "az": "aze",
    "kk": "kaz",
    "uz": "uzb",
    "id": "ind",
    "th": "tha",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build capped CPT JSONL files from corpus_download outputs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files or globs.")
    parser.add_argument("--output", type=Path, default=None, help="Single output JSONL for legacy mode.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for per-language JSONL outputs.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--target-tokens", type=int, default=10_000_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-files", action="store_true")
    parser.add_argument("--allow-missing-token-count", action="store_true")
    parser.add_argument(
        "--by-language",
        action="store_true",
        help="Write one JSONL per source language and a manifest for tagged Megatron datasets.",
    )
    parser.add_argument(
        "--allocation",
        choices=("proportional", "sequential"),
        default="proportional",
        help="How to cap per-language outputs when available tokens exceed --target-tokens.",
    )
    parser.add_argument(
        "--megatron-prefix-dir",
        type=Path,
        default=None,
        help="Directory where per-language Megatron indexed dataset prefixes will be written.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional HF tokenizer for rendering raw instruct messages with the model chat template.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Passed to AutoTokenizer when --tokenizer is used.",
    )
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


def get_language(record: dict[str, Any], path: Path) -> str:
    language = record.get("language")
    if isinstance(language, str) and language:
        return language
    stem = path.name
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    return stem.split(".")[0]


def get_vector_language(record: dict[str, Any], language: str) -> str:
    iso3 = record.get("iso3")
    if isinstance(iso3, str) and iso3:
        return iso3
    return DEFAULT_VECTOR_LANGUAGE_BY_LANGUAGE.get(language, language)


def flatten_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    flattened = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if content:
            flattened.append({"role": role, "content": content})
    return flattened


def load_tokenizer(tokenizer_name: str | None, trust_remote_code: bool) -> Any | None:
    if tokenizer_name is None:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers to render instruct messages with --tokenizer.") from exc
    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)


def render_text(record: dict[str, Any], tokenizer: Any | None) -> str:
    messages = flatten_messages(record.get("messages")) or flatten_messages(record.get("conversations"))
    if messages and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)

    text = record.get("text")
    if isinstance(text, str):
        return text.strip()

    if messages:
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    prompt = record.get("prompt") or record.get("instruction") or record.get("input")
    response = record.get("response") or record.get("output") or record.get("completion")
    if prompt and response:
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "user", "content": str(prompt).strip()},
                {"role": "assistant", "content": str(response).strip()},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)
        return f"user: {prompt}\nassistant: {response}"
    return ""


def scan_available_tokens(
    paths: list[Path],
    allow_missing_token_count: bool,
) -> tuple[dict[str, int], dict[str, str], dict[str, int]]:
    available_by_language: dict[str, int] = defaultdict(int)
    vector_language_by_language: dict[str, str] = {}
    docs_by_language: dict[str, int] = defaultdict(int)

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                language = get_language(record, path)
                try:
                    token_count = get_token_count(record, allow_missing_token_count)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                available_by_language[language] += token_count
                docs_by_language[language] += 1
                vector_language_by_language.setdefault(language, get_vector_language(record, language))

    return dict(available_by_language), vector_language_by_language, dict(docs_by_language)


def allocate_targets(available_by_language: dict[str, int], target_tokens: int, allocation: str) -> dict[str, int]:
    total_available = sum(available_by_language.values())
    if total_available <= target_tokens:
        return dict(available_by_language)

    if allocation == "sequential":
        remaining = target_tokens
        targets = {}
        for language, available in available_by_language.items():
            take = min(available, remaining)
            targets[language] = take
            remaining -= take
        return targets

    languages = sorted(available_by_language)
    raw_targets = {
        language: int(available_by_language[language] * target_tokens / total_available)
        for language in languages
    }
    remainder = target_tokens - sum(raw_targets.values())
    by_fraction = sorted(
        languages,
        key=lambda language: (available_by_language[language] * target_tokens / total_available) % 1,
        reverse=True,
    )
    for language in by_fraction[:remainder]:
        raw_targets[language] += 1
    return raw_targets


def write_legacy_output(args: argparse.Namespace, paths: list[Path], tokenizer: Any | None) -> None:
    if args.output is None:
        raise SystemExit("--output is required unless --by-language is used.")

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
                    text = render_text(record, tokenizer)
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
        "mode": "single",
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


def write_by_language_outputs(args: argparse.Namespace, paths: list[Path], tokenizer: Any | None) -> None:
    if args.output_dir is None:
        raise SystemExit("--output-dir is required with --by-language.")

    if args.shuffle_files:
        random.Random(args.seed).shuffle(paths)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (output_dir / "language_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    available_by_language, vector_language_by_language, scanned_docs_by_language = scan_available_tokens(
        paths,
        args.allow_missing_token_count,
    )
    targets_by_language = allocate_targets(available_by_language, args.target_tokens, args.allocation)
    vector_languages = sorted(set(vector_language_by_language.values()))
    vector_language_to_id = {language: idx for idx, language in enumerate(vector_languages)}

    handles = {}
    written_tokens: dict[str, int] = defaultdict(int)
    written_docs: dict[str, int] = defaultdict(int)
    try:
        for language in sorted(available_by_language):
            handles[language] = (output_dir / f"{language}.jsonl").open("w", encoding="utf-8")

        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    record = json.loads(line)
                    language = get_language(record, path)
                    target = targets_by_language.get(language, 0)
                    if target <= 0 or written_tokens[language] >= target:
                        continue

                    text = render_text(record, tokenizer)
                    if not text:
                        continue
                    try:
                        token_count = get_token_count(record, args.allow_missing_token_count)
                    except ValueError as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc
                    if token_count and written_tokens[language] + token_count > target:
                        continue

                    handles[language].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written_tokens[language] += token_count
                    written_docs[language] += 1
    finally:
        for handle in handles.values():
            handle.close()

    languages = []
    for language in sorted(available_by_language):
        vector_language = vector_language_by_language[language]
        output_path = output_dir / f"{language}.jsonl"
        megatron_output_prefix = None
        megatron_data_prefix = None
        if args.megatron_prefix_dir is not None:
            megatron_output_prefix = args.megatron_prefix_dir / language / f"qwen3_cpt_{language}"
            megatron_data_prefix = Path(str(megatron_output_prefix) + "_text_document")

        languages.append(
            {
                "language": language,
                "vector_language": vector_language,
                "language_id": vector_language_to_id[vector_language],
                "jsonl_path": str(output_path),
                "megatron_output_prefix": str(megatron_output_prefix) if megatron_output_prefix else None,
                "megatron_prefix": str(megatron_data_prefix) if megatron_data_prefix else None,
                "available_tokens": available_by_language[language],
                "target_tokens": targets_by_language.get(language, 0),
                "written_tokens": written_tokens[language],
                "available_docs": scanned_docs_by_language.get(language, 0),
                "written_docs": written_docs[language],
            }
        )
        print(
            f"{language}: wrote {written_docs[language]:,} docs / "
            f"{written_tokens[language]:,} tokens "
            f"(target={targets_by_language.get(language, 0):,}, available={available_by_language[language]:,})"
        )

    total_tokens = sum(written_tokens.values())
    total_docs = sum(written_docs.values())
    manifest = {
        "mode": "by_language",
        "target_tokens": args.target_tokens,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "allocation": args.allocation,
        "language_id_to_vector_language": {
            str(idx): language for language, idx in vector_language_to_id.items()
        },
        "languages": languages,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote {len(languages)} language JSONL files to {output_dir}")
    print(f"Wrote {total_docs:,} docs / {total_tokens:,} tokens")
    if total_tokens < args.target_tokens:
        print(
            f"[WARN] Target was {args.target_tokens:,} tokens, "
            f"but only {total_tokens:,} tokens were written."
        )
    print(f"Wrote manifest to {manifest_path}")


def main() -> None:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    tokenizer = load_tokenizer(args.tokenizer, args.trust_remote_code)

    if args.by_language:
        write_by_language_outputs(args, paths, tokenizer)
    else:
        write_legacy_output(args, paths, tokenizer)


if __name__ == "__main__":
    main()
