#!/usr/bin/env python3
"""Independently validate semantic-memory SciFact raw and aggregate receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROW_SCHEMA = "semantic-memory-scifact-query-v1"
AGGREGATE_SCHEMA = "semantic-memory-scifact-aggregate-v1"
SCHEMA_DESCRIPTOR = (
    "semantic-memory-scifact-eval-v1|split=lowest-sha256-query-id|"
    "metrics=ndcg10,recall1,5,10,mrr10,map10,success1,5,10|"
    "latency=mean,p50,p95,max|rows=jsonl"
)


class ValidationError(RuntimeError):
    pass


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


def close(actual: object, expected: object, path: str = "value") -> None:
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9):
            raise ValidationError(f"{path} mismatch: {actual!r} != {expected!r}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValidationError(f"{path} keys mismatch: {set(actual) if isinstance(actual, dict) else actual!r} != {set(expected)}")
        for key, value in expected.items():
            close(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValidationError(f"{path} list mismatch")
        for index, value in enumerate(expected):
            close(actual[index], value, f"{path}[{index}]")
    elif actual != expected:
        raise ValidationError(f"{path} mismatch: {actual!r} != {expected!r}")


def selected_query_ids(all_ids: list[str], split: str, calibration_count: int) -> set[str]:
    ordered = sorted(all_ids, key=lambda query_id: (hashlib.sha256(query_id.encode("utf-8")).hexdigest(), query_id))
    calibration = set(ordered[:calibration_count])
    if split == "calibration":
        return calibration
    if split == "heldout":
        return set(ordered) - calibration
    if split == "all":
        return set(ordered)
    raise ValidationError(f"unknown split: {split}")


def query_metrics(ranked: list[str], qrels: dict[str, int]) -> dict[str, float]:
    relevant = {doc_id for doc_id, grade in qrels.items() if int(grade) > 0}
    if not relevant:
        raise ValidationError("query has no positive qrels")

    def recall(k: int) -> float:
        return len(set(ranked[:k]) & relevant) / len(relevant)

    def success(k: int) -> float:
        return float(bool(set(ranked[:k]) & relevant))

    dcg = sum((2.0 ** int(qrels.get(doc_id, 0)) - 1.0) / math.log2(rank + 1.0)
              for rank, doc_id in enumerate(ranked[:10], 1))
    ideal = sorted((int(value) for value in qrels.values() if int(value) > 0), reverse=True)[:10]
    idcg = sum((2.0 ** grade - 1.0) / math.log2(rank + 1.0) for rank, grade in enumerate(ideal, 1))
    first = next((rank for rank, doc_id in enumerate(ranked[:10], 1) if doc_id in relevant), None)
    precision_sum = 0.0
    hits = 0
    for rank, doc_id in enumerate(ranked[:10], 1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return {
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "recall_at_1": recall(1), "recall_at_5": recall(5), "recall_at_10": recall(10),
        "mrr_at_10": 1.0 / first if first else 0.0,
        "map_at_10": precision_sum / min(len(relevant), 10),
        "success_at_1": success(1), "success_at_5": success(5), "success_at_10": success(10),
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def aggregate_metrics(rows: list[dict]) -> dict:
    metric_names = list(query_metrics([], {"fixture": 1}))
    sums = {name: 0.0 for name in metric_names}
    latencies: list[float] = []
    result_counts: Counter[int] = Counter()
    unique_docs: set[str] = set()
    top_ones: Counter[str] = Counter()
    failures = 0
    for row in rows:
        ranked = [str(doc_id) for doc_id in row["ranked_doc_ids"]]
        metrics = query_metrics(ranked, {str(key): int(value) for key, value in row["qrels"].items()})
        if "metrics" in row:
            close(row["metrics"], metrics, f"row[{row['query_id']}].metrics")
        for name, value in metrics.items():
            sums[name] += value
        latency = float(row["latency_ms"])
        if not math.isfinite(latency) or latency < 0:
            raise ValidationError(f"invalid latency for {row['query_id']}: {latency}")
        latencies.append(latency)
        result_counts[len(ranked)] += 1
        unique_docs.update(ranked)
        if ranked:
            top_ones[ranked[0]] += 1
        if row.get("error") is not None:
            failures += 1
    count = len(rows)
    if count == 0:
        raise ValidationError("receipt contains no query rows")
    most_common = sorted(top_ones.items(), key=lambda item: (-item[1], item[0]))[:1]
    top_doc, top_count = most_common[0] if most_common else (None, 0)
    return {
        **{name: sums[name] / count for name in metric_names},
        "latency_ms": {
            "mean": sum(latencies) / count, "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95), "max": max(latencies),
        },
        "failures": failures,
        "result_count_distribution": {str(key): result_counts[key] for key in sorted(result_counts)},
        "unique_retrieved_docs": len(unique_docs),
        "repeated_top1": {
            "doc_id": top_doc, "count": top_count,
            "frequency": top_count / sum(top_ones.values()) if top_ones else 0.0,
        },
    }


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if row.get("schema") != ROW_SCHEMA:
                raise ValidationError(f"row {line_number} has unexpected schema")
            rows.append(row)
    return rows


def validate(aggregate_path: Path, rows_path: Path | None = None) -> None:
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("schema") != AGGREGATE_SCHEMA:
        raise ValidationError("unexpected aggregate schema")
    receipt = aggregate["receipt"]
    if receipt["schema_sha256"] != sha256_bytes(SCHEMA_DESCRIPTOR.encode("utf-8")):
        raise ValidationError("schema hash mismatch")
    if rows_path is None:
        candidate = Path(aggregate["per_query_path"])
        rows_path = candidate if candidate.is_absolute() else aggregate_path.parent / candidate
    rows_path = rows_path.resolve()
    rows = read_rows(rows_path)
    if len(rows) != int(aggregate["row_count"]):
        raise ValidationError(f"row count mismatch: {len(rows)} != {aggregate['row_count']}")
    if sha256_file(rows_path) != aggregate["per_query_sha256"]:
        raise ValidationError("per-query JSONL hash mismatch")

    corpus_path = Path(receipt["corpus"]["path"])
    if not corpus_path.is_absolute():
        corpus_path = aggregate_path.parent / corpus_path
    corpus_path = corpus_path.resolve()
    if sha256_file(corpus_path) != receipt["corpus"]["file_sha256"]:
        raise ValidationError("corpus file hash mismatch")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    close(receipt["corpus"]["source_hashes"], corpus["source_hashes"], "source_hashes")
    close(receipt["corpus"]["payload_hashes"], corpus["payload_hashes"], "payload_hashes")

    documents_base = [{key: row[key] for key in ("doc_id", "title", "text", "semantic_text")} for row in corpus["documents"]]
    queries_base = [{"query_id": row["query_id"], "text": row["text"]} for row in corpus["queries"]]
    qrels_payload = {row["query_id"]: row["qrels"] for row in corpus["queries"]}
    recomputed_payload_hashes = {
        "corpus_sha256": sha256_bytes(canonical_bytes(documents_base)),
        "query_sha256": sha256_bytes(canonical_bytes(queries_base)),
        "qrels_sha256": sha256_bytes(canonical_bytes(qrels_payload)),
    }
    close(corpus["payload_hashes"], recomputed_payload_hashes, "corpus.payload_hashes")

    query_map = {str(row["query_id"]): row for row in corpus["queries"]}
    split = receipt["split_definition"]
    expected_ids = selected_query_ids(list(query_map), aggregate["split"], int(split["calibration_count"]))
    actual_ids = [str(row["query_id"]) for row in rows]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValidationError("duplicate query IDs in raw rows")
    if set(actual_ids) != expected_ids:
        raise ValidationError("split membership mismatch")
    selected_hash = sha256_bytes("\n".join(actual_ids).encode("utf-8"))
    if selected_hash != split["selected_query_ids_sha256"]:
        raise ValidationError("selected query ID hash mismatch")
    for row in rows:
        query = query_map[row["query_id"]]
        if row["mode"] != aggregate["mode"] or row["split"] != aggregate["split"]:
            raise ValidationError(f"mode/split mismatch for {row['query_id']}")
        if row["query_sha256"] != sha256_bytes(query["text"].encode("utf-8")):
            raise ValidationError(f"query hash mismatch for {row['query_id']}")
        close(row["qrels"], query["qrels"], f"qrels[{row['query_id']}]")
        result_ids = [result["doc_id"] for result in row["results"]]
        if result_ids != row["ranked_doc_ids"]:
            raise ValidationError(f"ranked IDs/results mismatch for {row['query_id']}")
        if len(result_ids) != len(set(result_ids)):
            raise ValidationError(f"duplicate ranked document for {row['query_id']}")
        if [result["rank"] for result in row["results"]] != list(range(1, len(result_ids) + 1)):
            raise ValidationError(f"non-contiguous ranks for {row['query_id']}")

    close(aggregate["metrics"], aggregate_metrics(rows), "aggregate.metrics")
    executable = receipt["executable"]
    executable_path = Path(executable["path"])
    if executable_path.is_file() and sha256_file(executable_path) != executable["sha256"]:
        raise ValidationError("current executable hash mismatch")
    runner_source = receipt.get("source", {}).get("runner_path")
    if runner_source and Path(runner_source).is_file():
        if sha256_file(Path(runner_source)) != receipt["source"]["runner_sha256"]:
            raise ValidationError("runner source hash mismatch")
    mapping_path = receipt.get("store", {}).get("mapping_path")
    if mapping_path and Path(mapping_path).is_file():
        if sha256_file(Path(mapping_path)) != receipt["store"]["mapping_sha256"]:
            raise ValidationError("store mapping hash mismatch")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = {
            "source_hashes": {"corpus_sha256": "sha256:source-c", "query_sha256": "sha256:source-q", "qrels_sha256": "sha256:source-r"},
            "documents": [
                {"doc_id": "d1", "title": "", "text": "one", "semantic_text": "one", "embedding": [1.0, 0.0]},
                {"doc_id": "d2", "title": "", "text": "two", "semantic_text": "two", "embedding": [0.0, 1.0]},
            ],
            "queries": [
                {"query_id": "q1", "text": "one", "embedding": [1.0, 0.0], "qrels": {"d1": 1}},
                {"query_id": "q2", "text": "two", "embedding": [0.0, 1.0], "qrels": {"d2": 1}},
            ],
        }
        documents_base = [{key: row[key] for key in ("doc_id", "title", "text", "semantic_text")} for row in corpus["documents"]]
        queries_base = [{"query_id": row["query_id"], "text": row["text"]} for row in corpus["queries"]]
        qrels = {row["query_id"]: row["qrels"] for row in corpus["queries"]}
        corpus["payload_hashes"] = {
            "corpus_sha256": sha256_bytes(canonical_bytes(documents_base)),
            "query_sha256": sha256_bytes(canonical_bytes(queries_base)),
            "qrels_sha256": sha256_bytes(canonical_bytes(qrels)),
        }
        corpus_path = root / "corpus.json"
        corpus_path.write_text(json.dumps(corpus, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        rows = []
        for query_id, text, relevant, ranked, latency in (
            ("q1", "one", "d1", ["d1", "d2"], 1.0),
            ("q2", "two", "d2", ["d1", "d2"], 3.0),
        ):
            rows.append({
                "schema": ROW_SCHEMA, "mode": "vector_only", "split": "all", "query_id": query_id,
                "query_sha256": sha256_bytes(text.encode()), "qrels": {relevant: 1}, "ranked_doc_ids": ranked,
                "results": [{"rank": index, "doc_id": doc_id} for index, doc_id in enumerate(ranked, 1)],
                "latency_ms": latency, "error": None, "metrics": query_metrics(ranked, {relevant: 1}),
            })
        rows_path = root / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        metrics = aggregate_metrics(rows)
        assert math.isclose(metrics["mrr_at_10"], 0.75)
        assert metrics["latency_ms"] == {"mean": 2.0, "p50": 1.0, "p95": 3.0, "max": 3.0}
        aggregate = {
            "schema": AGGREGATE_SCHEMA, "mode": "vector_only", "split": "all", "row_count": 2,
            "per_query_path": str(rows_path), "per_query_sha256": sha256_file(rows_path), "metrics": metrics,
            "receipt": {
                "schema_sha256": sha256_bytes(SCHEMA_DESCRIPTOR.encode()),
                "corpus": {"path": str(corpus_path), "file_sha256": sha256_file(corpus_path),
                           "source_hashes": corpus["source_hashes"], "payload_hashes": corpus["payload_hashes"]},
                "split_definition": {
                    "calibration_count": 1,
                    "selected_query_ids_sha256": sha256_bytes(b"q1\nq2"),
                },
                "executable": {"path": str(root / "missing-runner"), "sha256": "sha256:not-present"},
            },
        }
        aggregate_path = root / "aggregate.json"
        aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
        validate(aggregate_path, rows_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", nargs="?", type=Path)
    parser.add_argument("per_query", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("SciFact receipt validator self-test: ok")
        elif args.aggregate:
            validate(args.aggregate.resolve(), args.per_query.resolve() if args.per_query else None)
            print("SciFact receipt validation: ok")
        else:
            parser.error("provide AGGREGATE [PER_QUERY] or --self-test")
    except (OSError, KeyError, TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        print(f"SciFact receipt validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
