import argparse
import hashlib
import json
import math
import time

from harness import POOL, RESULTS, configure, probe

p = argparse.ArgumentParser()
p.add_argument("--embedder", required=True)
p.add_argument("--qa-model", required=True)
p.add_argument("--k", type=int, required=True)
p.add_argument("--search", default="single_pass")
p.add_argument("--chunk-size", type=int, default=128)
p.add_argument("--batch", type=int, default=1)
p.add_argument("--low-rate", type=float, default=1.0)
p.add_argument("--vm-cores", type=int, default=48)
args = p.parse_args()

configure("embed", 1, 1, EMBEDDER=args.embedder, CHUNK_SIZE=args.chunk_size,
          BATCH_SIZE=args.batch)
configure("qa", 1, 1, QA_MODEL=args.qa_model, BATCH_SIZE=args.batch)

cursor = 0
probes = []


def run(tag, rate, slo=None):
    global cursor
    duration = max(40, math.ceil(100 / rate))
    result = probe(tag, rate, duration, cursor, args.k, args.search, slo)
    cursor += result["sent"] + int(rate * result["warmup"]) + 20
    probes.append({"tag": tag, "rate": rate, "p95_ms": result["p95_ms"],
                   "failed": result["failed"], "sent": result["sent"],
                   "achieved_rps": result["achieved_rps"]})
    print(probes[-1], flush=True)
    time.sleep(20)
    return result


low = run("low", args.low_rate)
l0 = low["p95_ms"]
slo = 2 * l0


def within(result):
    return result["p95_ms"] <= slo and result["failed"] <= 0.01 * result["sent"]


lo, hi = args.low_rate, None
rate = 2 * args.low_rate
while rate <= 512:
    result = run(f"ramp_{rate:g}", rate, slo)
    if within(result):
        lo = rate
        rate *= 2
    else:
        hi = rate
        break

if hi is not None:
    for step in range(4):
        mid = (lo + hi) / 2
        result = run(f"bisect_{step}_{mid:.2f}", mid, slo)
        if within(result):
            lo = mid
        else:
            hi = mid

t0 = lo
multiplier = 4
while multiplier > 1 and 2 * multiplier > 0.75 * args.vm_cores:
    multiplier -= 1

digest = hashlib.sha256(POOL.read_bytes()).hexdigest()
summary = {
    "reference": {"embedder": args.embedder, "qa_model": args.qa_model, "k": args.k,
                  "search": args.search, "chunk_size": args.chunk_size,
                  "batch": args.batch, "replicas": 1, "cores_per_stage": 1},
    "L0_ms": l0,
    "SLO_ms": slo,
    "T0_rps": t0,
    "multiplier": multiplier,
    "target_rps": multiplier * t0,
    "seed": 8001,
    "pool_sha256": digest,
    "probes": probes,
}
(RESULTS / "sla.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({key: summary[key] for key in
                  ["L0_ms", "SLO_ms", "T0_rps", "multiplier", "target_rps"]}, indent=2))