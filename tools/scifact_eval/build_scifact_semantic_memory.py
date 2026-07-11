#!/usr/bin/env python3
"""Build the official BEIR SciFact test corpus for semantic-memory evaluation.

The output contains the semantic text and caller-supplied dense vectors needed
by the Rust production-retrieval evaluator. Embeddings are cached in an
append-only, fsync'd JSONL file so interrupted runs resume without re-embedding.
This builder does not compute retrieval metrics or inspect benchmark outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

import requests

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
DEFAULT_MODEL = "all-minilm:latest"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
EXPECTED_DOCUMENTS = 5_183
EXPECTED_TEST_QUERIES = 300
SCHEMA = "semantic-memory-scifact-corpus-v1"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def download_and_extract(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "scifact.zip"
    partial_path = data_dir / "scifact.zip.part"
    extract_dir = data_dir / "scifact"
    if not zip_path.exists():
        log(f"download {SCIFACT_URL}")
        with urllib.request.urlopen(SCIFACT_URL) as response, partial_path.open("wb") as output:
            shutil.copyfileobj(response, output)
            output.flush()
            os.fsync(output.fileno())
        partial_path.replace(zip_path)

    required = (extract_dir / "corpus.jsonl", extract_dir / "queries.jsonl", extract_dir / "qrels" / "test.tsv")
    if not all(path.is_file() for path in required):
        log(f"extract {zip_path}")
        with zipfile.ZipFile(zip_path) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                parts = PurePosixPath(info.filename).parts
                if info.filename.startswith("/") or ".." in parts:
                    raise RuntimeError(f"unsafe zip member: {info.filename}")
                archive.extract(info, data_dir)
    if not all(path.is_file() for path in required):
        raise RuntimeError("SciFact archive did not contain corpus.jsonl, queries.jsonl, and qrels/test.tsv")
    return extract_dir


def load_test_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            raise RuntimeError(f"invalid qrels row {line_number}: {line!r}")
        query_id, doc_id, raw_score = fields[:3]
        score = int(raw_score)
        if score > 0:
            qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def truncate_utf8_chars(text: str, max_chars: int) -> str:
    """Truncate by Unicode scalar values; encoding the result is always valid UTF-8."""
    return text[:max_chars]


def document_text(row: dict, max_chars: int) -> str:
    title = str(row.get("title") or "").strip()
    body = str(row.get("text") or "").strip()
    return truncate_utf8_chars(f"{title}\n{body}".strip() if title else body, max_chars)


def normalize(vector: list[float]) -> list[float]:
    if not vector or any(not math.isfinite(value) for value in vector):
        raise RuntimeError("embedding is empty or contains non-finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise RuntimeError("embedding has zero norm")
    return [float(value / norm) for value in vector]


def cache_key(kind: str, item_id: str, model: str, text: str) -> str:
    material = canonical_bytes({"kind": kind, "id": item_id, "model": model, "text": text})
    return hashlib.sha256(material).hexdigest()


def load_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                embedding = [float(value) for value in row["embedding"]]
                if not embedding or any(not math.isfinite(value) for value in embedding):
                    raise ValueError("invalid embedding")
                cache[str(row["key"])] = embedding
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid cache row {path}:{line_number}: {exc}") from exc
    return cache


def append_cache(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def embed_one(session: requests.Session, base_url: str, model: str, prompt: str, timeout: float, retries: int) -> list[float]:
    last_error: Exception | None = None
    # Character truncation is only a proxy for the model's token limit. Most
    # inputs fit at the declared 700-char ceiling, but pathological tokenization
    # can still exceed all-minilm's context. Fall back deterministically and
    # disclose the policy in corpus metadata rather than dropping the row.
    fallback_lengths = list(dict.fromkeys([len(prompt), 500, 300, 120]))
    for max_chars in fallback_lengths:
        candidate = truncate_utf8_chars(prompt, min(len(prompt), max_chars))
        for attempt in range(1, retries + 1):
            try:
                response = session.post(
                    f"{base_url.rstrip('/')}/api/embeddings",
                    json={"model": model, "prompt": candidate},
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                embedding = payload.get("embedding")
                if not isinstance(embedding, list):
                    raise RuntimeError(f"missing embedding in Ollama response: {payload}")
                if len(candidate) < len(prompt):
                    log(f"embedding fallback: {len(prompt)} -> {len(candidate)} chars")
                return normalize([float(value) for value in embedding])
            except (requests.RequestException, RuntimeError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt != retries:
                    time.sleep(min(0.5 * (2 ** (attempt - 1)), 8.0))
    raise RuntimeError(
        f"embedding failed after {retries} attempts at each fallback length: {last_error}"
    )


def embed_items(
    items: Iterable[tuple[str, str, str]], *, cache: dict[str, list[float]], cache_path: Path,
    base_url: str, model: str, timeout: float, retries: int, label: str,
) -> dict[str, list[float]]:
    rows = list(items)
    output: dict[str, list[float]] = {}
    session = requests.Session()
    started = time.monotonic()
    dimensions: int | None = None
    for index, (item_id, key, text) in enumerate(rows, 1):
        embedding = cache.get(key)
        if embedding is None:
            embedding = embed_one(session, base_url, model, text, timeout, retries)
            append_cache(cache_path, {
                "schema": "semantic-memory-scifact-embedding-cache-v1", "key": key,
                "item_id": item_id, "model": model, "text_sha256": sha256_bytes(text.encode("utf-8")),
                "dimensions": len(embedding), "embedding": embedding,
            })
            cache[key] = embedding
        dimensions = dimensions or len(embedding)
        if len(embedding) != dimensions:
            raise RuntimeError(f"embedding dimension mismatch for {item_id}: {len(embedding)} != {dimensions}")
        output[item_id] = embedding
        if index == 1 or index % 100 == 0 or index == len(rows):
            elapsed = max(time.monotonic() - started, 0.001)
            log(f"{label}: {index}/{len(rows)} ({index / elapsed:.2f}/s)")
    return output


def build(args: argparse.Namespace) -> None:
    if args.max_chars != 700:
        log(f"warning: nonstandard truncation requested: {args.max_chars} characters")
    out_path = Path(args.out).resolve()
    work_dir = Path(args.work_dir).resolve()
    scifact_dir = download_and_extract(work_dir / "data")
    corpus_path = scifact_dir / "corpus.jsonl"
    queries_path = scifact_dir / "queries.jsonl"
    qrels_path = scifact_dir / "qrels" / "test.tsv"
    corpus_rows = read_jsonl(corpus_path)
    all_query_rows = read_jsonl(queries_path)
    qrels = load_test_qrels(qrels_path)
    query_by_id = {str(row["_id"]): row for row in all_query_rows}
    query_ids = sorted(qrels)
    if len(corpus_rows) != EXPECTED_DOCUMENTS or len(query_ids) != EXPECTED_TEST_QUERIES:
        raise RuntimeError(
            f"official SciFact shape mismatch: docs={len(corpus_rows)} queries={len(query_ids)}; "
            f"expected {EXPECTED_DOCUMENTS}/{EXPECTED_TEST_QUERIES}"
        )
    missing_queries = [query_id for query_id in query_ids if query_id not in query_by_id]
    if missing_queries:
        raise RuntimeError(f"qrels reference missing queries: {missing_queries[:5]}")

    documents_base = [{
        "doc_id": str(row["_id"]), "title": str(row.get("title") or ""),
        "text": str(row.get("text") or ""), "semantic_text": document_text(row, args.max_chars),
    } for row in corpus_rows]
    queries_base = [{
        "query_id": query_id,
        "text": truncate_utf8_chars(str(query_by_id[query_id].get("text") or ""), args.max_chars),
        "qrels": dict(sorted(qrels[query_id].items())),
    } for query_id in query_ids]

    cache_path = work_dir / f"embeddings-{args.model.replace(':', '-')}-{args.max_chars}.jsonl"
    cache = load_cache(cache_path)
    log(f"corpus_docs={len(documents_base)} test_queries={len(queries_base)} cache_entries={len(cache)}")
    doc_items = [
        (
            row["doc_id"],
            cache_key("doc", row["doc_id"], args.model, f"search_document: {row['semantic_text']}"),
            f"search_document: {row['semantic_text']}",
        )
        for row in documents_base
    ]
    query_items = [
        (
            row["query_id"],
            cache_key("query", row["query_id"], args.model, f"search_query: {row['text']}"),
            f"search_query: {row['text']}",
        )
        for row in queries_base
    ]
    doc_vectors = embed_items(doc_items, cache=cache, cache_path=cache_path, base_url=args.ollama_url,
                              model=args.model, timeout=args.timeout, retries=args.retries, label="docs")
    query_vectors = embed_items(query_items, cache=cache, cache_path=cache_path, base_url=args.ollama_url,
                                model=args.model, timeout=args.timeout, retries=args.retries, label="queries")
    dimensions = len(next(iter(doc_vectors.values())))
    if any(len(vector) != dimensions for vector in query_vectors.values()):
        raise RuntimeError("document/query embedding dimensions differ")
    documents = [{**row, "embedding": doc_vectors[row["doc_id"]]} for row in documents_base]
    queries = [{**row, "embedding": query_vectors[row["query_id"]]} for row in queries_base]
    canonical_qrels = {row["query_id"]: row["qrels"] for row in queries_base}
    payload = {
        "schema": SCHEMA, "corpus_id": "beir-scifact-test-v1",
        "source": {"url": SCIFACT_URL, "archive_sha256": sha256_file(work_dir / "data" / "scifact.zip")},
        "source_hashes": {
            "corpus_sha256": sha256_file(corpus_path), "query_sha256": sha256_file(queries_path),
            "qrels_sha256": sha256_file(qrels_path),
        },
        "payload_hashes": {
            "corpus_sha256": sha256_bytes(canonical_bytes(documents_base)),
            "query_sha256": sha256_bytes(canonical_bytes([{"query_id": row["query_id"], "text": row["text"]} for row in queries_base])),
            "qrels_sha256": sha256_bytes(canonical_bytes(canonical_qrels)),
        },
        "embedding": {
            "model": args.model,
            "dimensions": dimensions,
            "normalized": True,
            "document_prefix": "search_document: ",
            "query_prefix": "search_query: ",
            "prefix_owner": "semantic-memory production embedding seam",
        },
        "truncation": {
            "unit": "unicode_scalar_values",
            "max_chars": args.max_chars,
            "utf8_safe": True,
            "embedding_fallback_char_limits": [500, 300, 120],
            "note": "Fallback is used only when the embedding service rejects the declared max-char prompt due to effective token context.",
        },
        "counts": {"documents": len(documents), "test_queries": len(queries)},
        "documents": documents, "queries": queries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(out_path)
    log(f"wrote {out_path} bytes={out_path.stat().st_size} sha256={sha256_file(out_path)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="semantic-memory/target/scifact-eval/scifact-all-minilm-corpus.json")
    parser.add_argument("--work-dir", default="semantic-memory/target/scifact-eval/build")
    parser.add_argument("--model", default=os.environ.get("SCIFACT_EMBED_MODEL", DEFAULT_MODEL))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--max-chars", type=int, default=700)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    if args.max_chars <= 0 or args.retries <= 0:
        parser.error("--max-chars and --retries must be positive")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
