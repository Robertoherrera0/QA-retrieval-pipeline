import json
import math
import time

from harness import CACHE, RESULTS, build_cache, configure, probe

CACHED = False
OUT = RESULTS / "profiling.jsonl"
SLA = RESULTS / "sla.json"

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
RESOURCES = [(1, 1), (1, 2), (2, 2), (4, 2)]
BATCHES = [1, 4, 8]
OTHER_REPLICAS = 6
OTHER_CORES = 2

while not SLA.exists():
    time.sleep(30)

sla = json.loads(SLA.read_text())
ref = sla["reference"]
slo = sla["SLO_ms"]
t0 = sla["T0_rps"]
k = ref["k"]
search = ref["search"]

STAGES = {
    "embed": {"variants": EMBEDDERS, "other": "qa"},
    "qa": {"variants": QA_MODELS, "other": "embed"},
}


def stage_env(stage, variant, batch):
    if stage == "embed":
        return {"EMBEDDER": variant, "CHUNK_SIZE": ref["chunk_size"], "BATCH_SIZE": batch}
    return {"QA_MODEL": variant, "BATCH_SIZE": batch}


def reference_variant(stage):
    return ref["embedder"] if stage == "embed" else ref["qa_model"]


done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        done.add((r["stage"], r["variant"], r["replicas"], r["cores"], r["batch"]))


def within(result):
    return result["p95_ms"] <= slo and result["failed"] <= 0.01 * result["sent"]


def find_max(tag, guess):
    probes = []

    def run(rate):
        duration = max(25, math.ceil(80 / rate))
        result = probe(f"{tag}_{rate:.2f}", rate, duration, 0, k, search, slo,
                       cached=CACHED)
        probes.append({"rate": rate, "p95_ms": result["p95_ms"],
                       "failed": result["failed"], "sent": result["sent"]})
        time.sleep(10)
        return result

    rate = guess
    first = run(rate)
    lo = hi = best = None
    if within(first):
        lo, best = rate, first
        while rate < 512:
            rate *= 2
            result = run(rate)
            if within(result):
                lo, best = rate, result
            else:
                hi = rate
                break
    else:
        hi = rate
        while rate > 0.25:
            rate /= 2
            result = run(rate)
            if within(result):
                lo, best = rate, result
                break
            hi = rate

    if best is None:
        return 0.0, None, probes
    if hi is not None:
        for _ in range(3):
            mid = (lo + hi) / 2
            result = run(mid)
            if within(result):
                lo, best = mid, result
            else:
                hi = mid
    return lo, best, probes


last_max = {}
last_cost = {}

for stage, spec in STAGES.items():
    other = spec["other"]
    configure(other, OTHER_REPLICAS, OTHER_CORES,
              **stage_env(other, reference_variant(other), ref["batch"]))
    CACHED = stage == "qa"
    if CACHED and not CACHE.exists():
        try:
            build_cache(k, search)
        except Exception as error:
            print("cache failed, QA stage uses the live embedding stage", repr(error),
                  flush=True)
            CACHED = False
    for replicas, cores in RESOURCES:
        cost = replicas * cores
        for variant in spec["variants"]:
            for batch in BATCHES:
                key = (stage, variant, replicas, cores, batch)
                if key in done:
                    continue
                if stage in last_max:
                    guess = max(0.5, last_max[stage] * cost / last_cost[stage])
                else:
                    guess = max(0.5, t0 * cost)
                tag = f"{stage}_{variant}_{replicas}x{cores}_b{batch}"
                result = None
                for attempt in range(3):
                    try:
                        configure(stage, replicas, cores, **stage_env(stage, variant, batch))
                        result = find_max(tag, guess)
                        break
                    except Exception as error:
                        print("failed", tag, attempt, repr(error), flush=True)
                        time.sleep(60)
                if result is None:
                    continue
                max_rps, best, probes = result
                if max_rps > 0:
                    last_max[stage], last_cost[stage] = max_rps, cost
                row = {
                    "stage": stage, "variant": variant, "batch": batch,
                    "replicas": replicas, "cores": cores, "cost": cost,
                    "slo_ms": slo, "k": k, "search": search,
                    "upstream": "cached" if CACHED else "live",
                    "max_rps_meeting_slo": max_rps, "at_max": best, "probes": probes,
                }
                with OUT.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(tag, max_rps, flush=True)