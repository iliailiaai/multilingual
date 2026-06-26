#!/usr/bin/env python3
"""Download capped HF corpora for the project's multilingual mix.

The script streams Hugging Face datasets, writes JSONL files with a `text`
field, and records exact token counts measured with the selected tokenizer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

load_dataset = None
AutoTokenizer = None


FINEWEB2_REPO = "HuggingFaceFW/fineweb-2"
FINEWEB_REPO = "HuggingFaceFW/fineweb"
SMOLTALK_REPO = "HuggingFaceTB/smoltalk"
DEFAULT_TOKENIZER = "Qwen/Qwen3-1.7B"

NON_ENGLISH_TOKEN_CAP = 350_000_000
ENGLISH_TOKEN_CAP = 700_000_000
INSTRUCT_TOKEN_CAP = 350_000_000

LANGUAGES = [
    {"key": "en", "name": "English", "iso3": "eng", "source": "fineweb"},
    {"key": "ja", "name": "Japanese", "iso3": "jpn", "fineweb2_config": "jpn_Jpan"},
    {"key": "fr", "name": "French", "iso3": "fra", "fineweb2_config": "fra_Latn"},
    {"key": "es", "name": "Spanish", "iso3": "spa", "fineweb2_config": "spa_Latn"},
    {"key": "zh", "name": "Chinese", "iso3": "cmn", "fineweb2_config": "cmn_Hans"},
    {"key": "de", "name": "German", "iso3": "deu", "fineweb2_config": "deu_Latn"},
    {"key": "nl", "name": "Dutch", "iso3": "nld", "fineweb2_config": "nld_Latn"},
    {"key": "ru", "name": "Russian", "iso3": "rus", "fineweb2_config": "rus_Cyrl"},
    {"key": "uk", "name": "Ukrainian", "iso3": "ukr", "fineweb2_config": "ukr_Cyrl"},
    {"key": "pl", "name": "Polish", "iso3": "pol", "fineweb2_config": "pol_Latn"},
    {"key": "cs", "name": "Czech", "iso3": "ces", "fineweb2_config": "ces_Latn"},
    {"key": "pt", "name": "Portuguese", "iso3": "por", "fineweb2_config": "por_Latn"},
    {"key": "it", "name": "Italian", "iso3": "ita", "fineweb2_config": "ita_Latn"},
    {"key": "ur", "name": "Urdu", "iso3": "urd", "fineweb2_config": "urd_Arab"},
    {"key": "fa", "name": "Persian", "iso3": "pes", "fineweb2_config": "pes_Arab"},
    {"key": "ga", "name": "Irish", "iso3": "gle", "fineweb2_config": "gle_Latn"},
    {"key": "cy", "name": "Welsh", "iso3": "cym", "fineweb2_config": "cym_Latn"},
    {"key": "ar", "name": "Arabic", "iso3": "arb", "fineweb2_config": "arb_Arab"},
    {"key": "he", "name": "Hebrew", "iso3": "heb", "fineweb2_config": "heb_Hebr"},
    {"key": "fi", "name": "Finnish", "iso3": "fin", "fineweb2_config": "fin_Latn"},
    {"key": "et", "name": "Estonian", "iso3": "est", "fineweb2_config": "est_Latn"},
    {"key": "hu", "name": "Hungarian", "iso3": "hun", "fineweb2_config": "hun_Latn"},
    {"key": "tr", "name": "Turkish", "iso3": "tur", "fineweb2_config": "tur_Latn"},
    {"key": "az", "name": "Azerbaijani", "iso3": "aze", "fineweb2_config": "azj_Latn"},
    {"key": "kk", "name": "Kazakh", "iso3": "kaz", "fineweb2_config": "kaz_Cyrl"},
    {"key": "uz", "name": "Uzbek", "iso3": "uzb", "fineweb2_config": "uzn_Latn"},
    {"key": "id", "name": "Indonesian", "iso3": "ind", "fineweb2_config": "ind_Latn"},
    {"key": "th", "name": "Thai", "iso3": "tha", "fineweb2_config": "tha_Thai"},
]


@dataclass
class CorpusSpec:
    key: str
    name: str
    repo: str
    max_tokens: int
    kind: str
    iso3: str | None = None
    data_files: str | None = None
    dataset_config: str | None = None
    hf_config: str | None = None


@dataclass
class CorpusResult:
    key: str
    name: str
    repo: str
    kind: str
    output_file: str
    max_tokens: int
    min_tokens: int
    token_count: int
    rows_written: int
    rows_seen: int
    skipped_empty: int
    skipped_too_large: int
    completed: bool
    warning: str | None
    error: str | None = None


def build_specs(include_instruct: bool) -> list[CorpusSpec]:
    specs: list[CorpusSpec] = []
    for language in LANGUAGES:
        if language["key"] == "en":
            specs.append(
                CorpusSpec(
                    key="en",
                    name="English",
                    iso3="eng",
                    repo=FINEWEB_REPO,
                    data_files=f"hf://datasets/{FINEWEB_REPO}/sample/10BT/*.parquet",
                    dataset_config="sample-10BT",
                    max_tokens=ENGLISH_TOKEN_CAP,
                    kind="pretrain",
                    hf_config="sample-10BT",
                )
            )
            continue

        config = language["fineweb2_config"]
        specs.append(
            CorpusSpec(
                key=language["key"],
                name=language["name"],
                iso3=language["iso3"],
                repo=FINEWEB2_REPO,
                data_files=f"hf://datasets/{FINEWEB2_REPO}/data/{config}/train/*.parquet",
                dataset_config=config,
                max_tokens=NON_ENGLISH_TOKEN_CAP,
                kind="pretrain",
                hf_config=config,
            )
        )

    if include_instruct:
        specs.append(
            CorpusSpec(
                key="en_instruct",
                name="English instruct",
                repo=SMOLTALK_REPO,
                dataset_config="all",
                max_tokens=INSTRUCT_TOKEN_CAP,
                kind="instruct",
                hf_config="all",
            )
        )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FineWeb/FineWeb2/SmolTalk corpora capped by tokenizer token count."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("corpus_download/data"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["all"],
        help="Language keys to download, e.g. en ja ru, or all. Use en_instruct for SmolTalk.",
    )
    parser.add_argument("--no-instruct", action="store_true", help="Do not download SmolTalk.")
    parser.add_argument(
        "--min-token-ratio",
        type=float,
        default=0.99,
        help="Warn if a corpus finishes below this fraction of its cap.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--add-special-tokens-to-count",
        action="store_true",
        help="Count tokenizer-added special tokens. Default counts the written text only.",
    )
    parser.add_argument("--max-doc-tokens", type=int, default=None)
    parser.add_argument(
        "--tokenize-batch-size",
        type=int,
        default=256,
        help="Number of documents to tokenize in one tokenizer call.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of corpora to download in parallel. Each worker handles one corpus at a time.",
    )
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--stream-open-retries", type=int, default=3)
    return parser.parse_args()


def import_dependencies() -> None:
    global AutoTokenizer, load_dataset
    try:
        from datasets import load_dataset as datasets_load_dataset
        from transformers import AutoTokenizer as transformers_auto_tokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install packages from corpus_download/requirements.txt "
            "before running this script."
        ) from exc

    load_dataset = datasets_load_dataset
    AutoTokenizer = transformers_auto_tokenizer


def setup_logging(output_dir: Path, log_name: str = "download_run.log") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_name
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    for logger_name in ("httpx", "httpcore", "huggingface_hub", "fsspec"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def load_stream(spec: CorpusSpec, trust_remote_code: bool) -> Iterable[dict[str, Any]]:
    if load_dataset is None:
        raise RuntimeError("Dependencies are not loaded.")

    if spec.kind == "instruct":
        return load_dataset(
            spec.repo,
            spec.dataset_config,
            split="train",
            streaming=True,
            trust_remote_code=trust_remote_code,
        )

    try:
        return load_dataset(
            "parquet",
            data_files=spec.data_files,
            split="train",
            streaming=True,
        )
    except ValueError as exc:
        if "data_files are invalid" not in str(exc) or spec.dataset_config is None:
            raise
        logging.warning(
            "%s: parquet path %s is not available; falling back to dataset config %s",
            spec.key,
            spec.data_files,
            spec.dataset_config,
        )
        return load_dataset(
            spec.repo,
            spec.dataset_config,
            split="train",
            streaming=True,
            trust_remote_code=trust_remote_code,
        )


def text_from_messages(messages: Any, tokenizer: Any) -> str:
    if not isinstance(messages, list):
        return ""

    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception as exc:  # noqa: BLE001 - keep downloads moving on unexpected row shape.
            logging.warning("Could not apply chat template, falling back to plain text: %s", exc)

    lines = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip()
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def extract_text(example: dict[str, Any], spec: CorpusSpec, tokenizer: Any) -> str:
    if spec.kind == "instruct":
        if "messages" in example:
            return text_from_messages(example["messages"], tokenizer)
        if "conversations" in example:
            return text_from_messages(example["conversations"], tokenizer)
        if "text" in example:
            return str(example["text"])
        prompt = example.get("prompt") or example.get("instruction") or example.get("input")
        response = example.get("response") or example.get("output") or example.get("completion")
        if prompt and response:
            return f"user: {prompt}\nassistant: {response}"
        return ""

    return str(example.get("text", ""))


def count_tokens(tokenizer: Any, text: str, add_special_tokens: bool) -> int:
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def count_tokens_batch(tokenizer: Any, texts: list[str], add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(
        texts,
        add_special_tokens=add_special_tokens,
        padding=False,
        truncation=False,
    )
    return [len(input_ids) for input_ids in encoded["input_ids"]]


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"rows_seen": 0, "rows_written": 0, "token_count": 0}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def write_summary(log_file: Path, results: list[CorpusResult], args: argparse.Namespace) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": args.tokenizer,
        "add_special_tokens_to_count": args.add_special_tokens_to_count,
        "tokenize_batch_size": args.tokenize_batch_size,
        "workers": args.workers,
        "stream_open_retries": args.stream_open_retries,
        "output_dir": str(args.output_dir),
        "total_tokens": sum(result.token_count for result in results),
        "total_rows": sum(result.rows_written for result in results),
        "corpora": [asdict(result) for result in results],
    }
    write_json(log_file, payload)


def select_specs(specs: list[CorpusSpec], requested: list[str]) -> list[CorpusSpec]:
    if [item.lower() for item in requested] == ["all"]:
        return specs

    aliases: dict[str, str] = {}
    for spec in specs:
        for alias in (spec.key, spec.name, spec.iso3, spec.hf_config):
            if alias:
                aliases[alias.lower()] = spec.key
    aliases["instruct"] = "en_instruct"
    aliases["english_instruct"] = "en_instruct"
    aliases["english-instruct"] = "en_instruct"

    selected_keys = set()
    unknown = []
    for item in requested:
        key = aliases.get(item.lower())
        if key is None:
            unknown.append(item)
        else:
            selected_keys.add(key)

    if unknown:
        raise SystemExit(f"Unknown language/corpus key(s): {', '.join(unknown)}")
    return [spec for spec in specs if spec.key in selected_keys]


def result_from_state(
    spec: CorpusSpec,
    args: argparse.Namespace,
    error: Exception | str,
) -> CorpusResult:
    state_file = args.output_dir / f"{spec.key}.state.json"
    state = read_state(state_file)
    warning = f"{spec.key} failed: {error}"
    return CorpusResult(
        key=spec.key,
        name=spec.name,
        repo=spec.repo,
        kind=spec.kind,
        output_file=str(args.output_dir / f"{spec.key}.jsonl"),
        max_tokens=spec.max_tokens,
        min_tokens=int(spec.max_tokens * args.min_token_ratio),
        token_count=int(state.get("token_count", 0)),
        rows_written=int(state.get("rows_written", 0)),
        rows_seen=int(state.get("rows_seen", 0)),
        skipped_empty=int(state.get("skipped_empty", 0)),
        skipped_too_large=int(state.get("skipped_too_large", 0)),
        completed=False,
        warning=warning,
        error=str(error),
    )


def download_corpus(
    spec: CorpusSpec,
    tokenizer: Any,
    args: argparse.Namespace,
    log_file: Path,
    results: list[CorpusResult],
    write_incremental_summary: bool = True,
) -> CorpusResult:
    output_file = args.output_dir / f"{spec.key}.jsonl"
    state_file = args.output_dir / f"{spec.key}.state.json"
    min_tokens = int(spec.max_tokens * args.min_token_ratio)

    if output_file.exists() and not args.overwrite and not args.resume:
        raise SystemExit(
            f"{output_file} already exists. Use --resume to continue or --overwrite to replace it."
        )

    if args.overwrite:
        output_file.unlink(missing_ok=True)
        state_file.unlink(missing_ok=True)

    state = read_state(state_file) if args.resume else {"rows_seen": 0, "rows_written": 0, "token_count": 0}
    rows_seen = int(state["rows_seen"])
    rows_written = int(state["rows_written"])
    token_count = int(state["token_count"])
    skipped_empty = int(state.get("skipped_empty", 0))
    skipped_too_large = int(state.get("skipped_too_large", 0))

    logging.info(
        "Starting %s from %s (%s), cap=%s tokens, already=%s tokens",
        spec.key,
        spec.repo,
        spec.hf_config or spec.data_files,
        f"{spec.max_tokens:,}",
        f"{token_count:,}",
    )

    stream = None
    for attempt in range(1, args.stream_open_retries + 1):
        try:
            stream = load_stream(spec, args.trust_remote_code)
            break
        except Exception:
            if attempt >= args.stream_open_retries:
                raise
            sleep_seconds = min(60, 5 * attempt)
            logging.exception(
                "%s: failed to open stream on attempt %s/%s; retrying in %ss",
                spec.key,
                attempt,
                args.stream_open_retries,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
    if stream is None:
        raise RuntimeError(f"{spec.key}: failed to open stream")
    mode = "a" if args.resume and output_file.exists() else "w"
    completed = False
    pending: list[tuple[int, str]] = []

    with output_file.open(mode, encoding="utf-8") as out:
        def flush_pending() -> bool:
            nonlocal completed
            nonlocal rows_written
            nonlocal skipped_too_large
            nonlocal token_count

            if not pending:
                return False

            batch = pending.copy()
            pending.clear()
            token_counts = count_tokens_batch(
                tokenizer,
                [text for _, text in batch],
                args.add_special_tokens_to_count,
            )

            for _, text, doc_tokens in zip(
                (source_index for source_index, _ in batch),
                (text for _, text in batch),
                token_counts,
            ):
                if args.max_doc_tokens is not None and doc_tokens > args.max_doc_tokens:
                    skipped_too_large += 1
                    continue
                if token_count + doc_tokens > spec.max_tokens:
                    completed = True
                    return True

                record = {
                    "text": text,
                    "source": spec.repo,
                    "source_config": spec.hf_config,
                    "language": spec.key,
                    "language_name": spec.name,
                    "iso3": spec.iso3,
                    "kind": spec.kind,
                    "tokenizer": args.tokenizer,
                    "token_count": doc_tokens,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                token_count += doc_tokens
                rows_written += 1

                if rows_written % args.log_every == 0:
                    logging.info(
                        "%s: %s tokens, %s rows written",
                        spec.key,
                        f"{token_count:,}",
                        f"{rows_written:,}",
                    )
                    write_json(
                        state_file,
                        {
                            "rows_seen": rows_seen,
                            "rows_written": rows_written,
                            "token_count": token_count,
                            "skipped_empty": skipped_empty,
                            "skipped_too_large": skipped_too_large,
                        },
                    )
            return False

        for source_index, example in enumerate(stream):
            if source_index < rows_seen:
                continue

            rows_seen = source_index + 1
            text = extract_text(example, spec, tokenizer).strip()
            if not text:
                skipped_empty += 1
                continue

            pending.append((source_index, text))
            if len(pending) >= args.tokenize_batch_size and flush_pending():
                break

        if not completed:
            flush_pending()

    warning = None
    if token_count < min_tokens:
        warning = (
            f"{spec.key} has {token_count:,} tokens, below requested minimum "
            f"{min_tokens:,} ({args.min_token_ratio:.0%} of cap)."
        )
        logging.warning(warning)

    result = CorpusResult(
        key=spec.key,
        name=spec.name,
        repo=spec.repo,
        kind=spec.kind,
        output_file=str(output_file),
        max_tokens=spec.max_tokens,
        min_tokens=min_tokens,
        token_count=token_count,
        rows_written=rows_written,
        rows_seen=rows_seen,
        skipped_empty=skipped_empty,
        skipped_too_large=skipped_too_large,
        completed=completed,
        warning=warning,
        error=None,
    )
    write_json(state_file, {**asdict(result), "rows_seen": rows_seen})
    results.append(result)
    if write_incremental_summary:
        write_summary(log_file, results, args)
    logging.info("%s done: %s tokens in %s rows", spec.key, f"{token_count:,}", f"{rows_written:,}")
    return result


def download_corpus_worker(spec: CorpusSpec, args: argparse.Namespace) -> CorpusResult:
    setup_logging(args.output_dir, log_name=f"{spec.key}.download_run.log")
    import_dependencies()
    if AutoTokenizer is None:
        raise RuntimeError("Dependencies are not loaded.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    return download_corpus(
        spec,
        tokenizer,
        args,
        args.output_dir / f"{spec.key}.summary.json",
        [],
        write_incremental_summary=False,
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.tokenize_batch_size < 1:
        raise SystemExit("--tokenize-batch-size must be >= 1")

    args.output_dir = args.output_dir.resolve()
    log_file = (args.log_file or (args.output_dir / "download_summary.json")).resolve()
    setup_logging(args.output_dir)

    specs = build_specs(include_instruct=not args.no_instruct)
    selected_specs = select_specs(specs, args.languages)

    import_dependencies()
    if AutoTokenizer is None:
        raise RuntimeError("Dependencies are not loaded.")

    results: list[CorpusResult] = []
    if args.workers <= 1:
        logging.info("Loading tokenizer: %s", args.tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
        for spec in selected_specs:
            try:
                download_corpus(spec, tokenizer, args, log_file, results)
            except Exception as exc:
                logging.exception("Corpus failed for %s; continuing with other corpora", spec.key)
                result = result_from_state(spec, args, exc)
                results.append(result)
                write_summary(log_file, results, args)
    else:
        max_workers = min(args.workers, len(selected_specs))
        logging.info("Downloading %s corpora with %s workers", len(selected_specs), max_workers)
        order = {spec.key: index for index, spec in enumerate(selected_specs)}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_corpus_worker, spec, args): spec for spec in selected_specs}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logging.exception("Worker failed for %s; continuing with other corpora", spec.key)
                    result = result_from_state(spec, args, exc)
                results.append(result)
                results.sort(key=lambda item: order[item.key])
                write_summary(log_file, results, args)
                logging.info(
                    "Worker finished %s: %s tokens in %s rows",
                    spec.key,
                    f"{result.token_count:,}",
                    f"{result.rows_written:,}",
                )

    write_summary(log_file, results, args)
    failed = [result for result in results if result.error]
    warnings = [result.warning for result in results if result.warning]
    if failed:
        logging.error(
            "Completed with %s failed corpus/corpora: %s. Re-run with --resume after fixing paths/network.",
            len(failed),
            ", ".join(result.key for result in failed),
        )
        raise SystemExit(1)
    elif warnings:
        logging.warning("Completed with %s corpus warning(s). See %s", len(warnings), log_file)
    else:
        logging.info("Completed without token-count warnings. See %s", log_file)


if __name__ == "__main__":
    main()
