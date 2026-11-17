import argparse
import json
import math
import time

from harness import RESULTS, configure, probe

EMBEDDERS = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "bge-small-en-v1.5",
]
QA_MODELS = [
    "tinyroberta-squad2",
    "roberta-base-squad2",
    "distilbert-base-cased-distilled-squad",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
]
TEST_POOL = "testslice.jsonl"
GOLD_CACHE = "gold_cache.jsonl"
E2E_OUT = RESULTS / "e2e.jsonl"
GOLD_OUT = RESULTS / "gold.jsonl"
BIG = (6, 3)

p = argparse.ArgumentParser()
p.add_argument("part", choices=["accuracy", "load", "loadbig", "gold"])
p.add_argument("--rate", type=float, default=3.0, help="requests per second for accuracy and gold runs")
p.add_argument("--load-seconds", type=int, default=120)
p.add_argument("--questions", type=int, default=1000)
args = p.parse_args()

sla = json.loads((RESULTS / "sla.json").read_text())
ref = sla["reference"]
slo = sla["SLO_ms"]
target = sla["target_rps"]
k = ref["k"]
search = ref["search"]
batch = ref["batch"]


def embed_env(embedder):
    return {"EMBEDDER": embedder, "CHUNK_SIZE": ref["chunk_size"], "BATCH_SIZE": batch}


def qa_env(qa_model):
    return {"QA_MODEL": qa_model, "BATCH_SIZE": batch}


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def accuracy_from(tag):
    records = [json.loads(line) for line in
               (RESULTS / "loadruns" / f"{tag}.requests.jsonl").read_text().splitlines()]
    done = [r for r in records if r["ok"] and r["index"] < args.questions]
    if not done:
        return None
    return {
        "f1": sum(r["f1"] for r in done) / len(done),
        "em": sum(r["em"] for r in done) / len(done),
        "answered": len(done),
    }


def run_test_slice(tag, cache_name=None):
    duration = math.ceil(args.questions / args.rate * 1.25) + 15
    probe(tag, args.rate, duration, 0, k, search, None, 5,
          pool_name=TEST_POOL, cache_name=cache_name)
    return accuracy_from(tag)


def cheapest_resources(stage, variant):
    rows = [r for r in read_rows(RESULTS / "profiling.jsonl")
            if r["stage"] == stage and r["variant"] == variant and r["batch"] == batch]
    if not rows:
        return BIG, False
    enough = [r for r in rows if r["max_rps_meeting_slo"] >= target]
    if enough:
        best = min(enough, key=lambda r: (r["cost"], r["replicas"]))
        return (best["replicas"], best["cores"]), True
    best = max(rows, key=lambda r: (r["max_rps_meeting_slo"], -r["cost"]))
    return (best["replicas"], best["cores"]), False


def accuracy_part():
    done = {(r["embedder"], r["qa_model"]) for r in read_rows(E2E_OUT) if r.get("part") == "accuracy"}
    for embedder in EMBEDDERS:
        for qa_model in QA_MODELS:
            if (embedder, qa_model) in done:
                continue
            configure("embed", *BIG, **embed_env(embedder))
            configure("qa", *BIG, **qa_env(qa_model))
            result = run_test_slice(f"e2e_acc_{embedder}_{qa_model}")
            append(E2E_OUT, {"part": "accuracy", "embedder": embedder, "qa_model": qa_model,
                             "questions": args.questions, **(result or {})})
            print("accuracy", embedder, qa_model, result, flush=True)
            time.sleep(10)


def load_part():
    done = {(r["embedder"], r["qa_model"]) for r in read_rows(E2E_OUT) if r.get("part") == "load"}
    for embedder in EMBEDDERS:
        for qa_model in QA_MODELS:
            if (embedder, qa_model) in done:
                continue
            (er, ec), embed_ok = cheapest_resources("embed", embedder)
            (qr, qc), qa_ok = cheapest_resources("qa", qa_model)
            configure("embed", er, ec, **embed_env(embedder))
            configure("qa", qr, qc, **qa_env(qa_model))
            tag = f"e2e_load_{embedder}_{qa_model}"
            result = probe(tag, target, args.load_seconds, 0, k, search, slo)
            cost = er * ec + qr * qc
            row = {
                "part": "load", "embedder": embedder, "qa_model": qa_model,
                "embed_replicas": er, "embed_cores": ec, "qa_replicas": qr, "qa_cores": qc,
                "cost": cost, "core_seconds": cost * args.load_seconds,
                "rate": target, "duration": args.load_seconds, "slo_ms": slo,
                "p50_ms": result["p50_ms"], "p95_ms": result["p95_ms"], "p99_ms": result["p99_ms"],
                "slo_attainment": result["slo_attainment"], "failed": result["failed"],
                "sent": result["sent"], "achieved_rps": result["achieved_rps"],
                "meets_sla": result["p95_ms"] <= slo and result["failed"] <= 0.01 * result["sent"],
                "profile_says_target_reachable": bool(embed_ok and qa_ok),
            }
            append(E2E_OUT, row)
            print("load", embedder, qa_model, row["p95_ms"], row["meets_sla"], flush=True)
            time.sleep(10)


def load_big_part():
    rows = [r for r in read_rows(E2E_OUT) if r.get("part") == "load"]
    scaled = {(r["embedder"], r["qa_model"]) for r in rows if r.get("scaled")}
    for old in rows:
        embedder, qa_model = old["embedder"], old["qa_model"]
        if old.get("scaled") or old["profile_says_target_reachable"] or (embedder, qa_model) in scaled:
            continue
        (er, ec), (qr, qc) = BIG, BIG
        configure("embed", er, ec, **embed_env(embedder))
        configure("qa", qr, qc, **qa_env(qa_model))
        tag = f"e2e_loadbig_{embedder}_{qa_model}"
        result = probe(tag, target, args.load_seconds, 0, k, search, slo)
        cost = er * ec + qr * qc
        row = {
            "part": "load", "scaled": True, "embedder": embedder, "qa_model": qa_model,
            "embed_replicas": er, "embed_cores": ec, "qa_replicas": qr, "qa_cores": qc,
            "cost": cost, "core_seconds": cost * args.load_seconds,
            "rate": target, "duration": args.load_seconds, "slo_ms": slo,
            "p50_ms": result["p50_ms"], "p95_ms": result["p95_ms"], "p99_ms": result["p99_ms"],
            "slo_attainment": result["slo_attainment"], "failed": result["failed"],
            "sent": result["sent"], "achieved_rps": result["achieved_rps"],
            "meets_sla": result["p95_ms"] <= slo and result["failed"] <= 0.01 * result["sent"],
            "profile_says_target_reachable": False,
        }
        append(E2E_OUT, row)
        print("loadbig", embedder, qa_model, row["p95_ms"], row["meets_sla"], flush=True)
        time.sleep(10)


def build_gold_cache():
    lines = (RESULTS / TEST_POOL).read_text(encoding="utf-8").splitlines()
    with (RESULTS / GOLD_CACHE).open("w", encoding="utf-8") as f:
        for line in lines[:args.questions]:
            item = json.loads(line)
            gold = set(item["gold_titles"])
            chunks = []
            for number, paragraph in enumerate(item["paragraphs"]):
                if paragraph["title"] in gold:
                    chunks.append({
                        "rank": len(chunks), "text": paragraph["text"], "para_id": number,
                        "title": paragraph["title"], "start_index": 0,
                        "end_index": len(paragraph["text"]), "chunk_id": f"gold_{number}",
                        "distance": 0.0, "hop": 1,
                    })
            f.write(json.dumps({"id": item["id"], "chunks": chunks}) + "\n")


def gold_part():
    build_gold_cache()
    done = {r["qa_model"] for r in read_rows(GOLD_OUT)}
    for qa_model in QA_MODELS:
        if qa_model in done:
            continue
        configure("qa", *BIG, **qa_env(qa_model))
        result = run_test_slice(f"gold_{qa_model}", cache_name=GOLD_CACHE)
        append(GOLD_OUT, {"qa_model": qa_model, "questions": args.questions, **(result or {})})
        print("gold", qa_model, result, flush=True)
        time.sleep(10)


{"accuracy": accuracy_part, "load": load_part, "loadbig": load_big_part, "gold": gold_part}[args.part]()