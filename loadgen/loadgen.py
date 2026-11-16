import argparse
import asyncio
import json
import re
import string
import time
from collections import Counter

import grpc
import numpy as np

import qa_pb2
import qa_pb2_grpc

ROUND_ROBIN = json.dumps({"loadBalancingConfig": [{"round_robin": {}}]})


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score(prediction, reference):
    p = normalize_answer(prediction).split()
    g = normalize_answer(reference).split()
    if not p or not g:
        f1 = float(p == g)
    else:
        f1 = 2 * sum((Counter(p) & Counter(g)).values()) / (len(p) + len(g))
    return float(p == g), f1


def channel(target):
    return grpc.aio.insecure_channel(
        target, options=[("grpc.service_config", ROUND_ROBIN)]
    )


def load_pool(path, offset, count):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [json.loads(lines[(offset + i) % len(lines)]) for i in range(count)]


def summarize(args, arrivals, records):
    window = [i for i, due in enumerate(arrivals) if due >= args.warmup]
    rows = [records.get(i, {"ok": False}) for i in window]
    ok = [r for r in rows if r["ok"]]
    latency = np.array([r["e2e_ms"] if r["ok"] else args.timeout * 1000 for r in rows])

    def mean(values):
        return float(np.mean(values)) if len(values) else None

    p50, p95, p99 = (float(x) for x in np.percentile(latency, [50, 95, 99]))
    return {
        "rate": args.rate,
        "duration": args.duration,
        "warmup": args.warmup,
        "offset": args.offset,
        "seed": args.seed,
        "k": args.k,
        "search": args.search,
        "sent": len(rows),
        "failed": len(rows) - len(ok),
        "achieved_rps": len(ok) / (args.duration - args.warmup),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "slo_attainment": float(np.mean(latency <= args.slo_ms)) if args.slo_ms else None,
        "f1": mean([r["f1"] for r in ok]),
        "em": mean([r["em"] for r in ok]),
        "lag_ms": mean([r["lag_ms"] for r in ok]),
        "embed_queue_ms": mean([r["embed_queue_ms"] for r in ok]),
        "embed_service_ms": mean([r["embed_service_ms"] for r in ok]),
        "embed_network_ms": mean([
            r["embed_rtt_ms"] - r["embed_queue_ms"] - r["embed_service_ms"] for r in ok
        ]),
        "qa_queue_ms": mean([r["qa_queue_ms"] for r in ok]),
        "qa_service_ms": mean([r["qa_service_ms"] for r in ok]),
        "qa_network_ms": mean([
            r["qa_rtt_ms"] - r["qa_queue_ms"] - r["qa_service_ms"] for r in ok
        ]),
    }


async def run(args):
    rng = np.random.default_rng(args.seed)
    gaps = rng.exponential(1 / args.rate, size=int(args.rate * args.duration * 3) + 100)
    arrivals = np.cumsum(gaps)
    arrivals = arrivals[arrivals < args.duration]
    items = load_pool(args.pool, args.offset, len(arrivals))
    requests = [
        qa_pb2.RetrieveRequest(
            question=item["question"],
            paragraphs=[
                qa_pb2.Paragraph(title=p["title"], para_id=n, text=p["text"])
                for n, p in enumerate(item["paragraphs"])
            ],
            k=args.k,
            search=args.search,
        )
        for item in items
    ]

    embed = qa_pb2_grpc.EmbeddingStub(channel(args.embed))
    qa = qa_pb2_grpc.QAStub(channel(args.qa))
    records = {}

    async def one(index, due):
        record = {"index": index, "due": float(due), "question_id": items[index]["id"]}
        try:
            began = time.perf_counter()
            found = await embed.Retrieve(requests[index], timeout=args.timeout)
            middle = time.perf_counter()
            answer = await qa.Answer(
                qa_pb2.AnswerRequest(
                    question=items[index]["question"],
                    chunks=found.chunks,
                    allow_empty=args.allow_empty,
                ),
                timeout=args.timeout,
            )
            end = time.perf_counter()
        except Exception as error:
            record.update(ok=False, error=type(error).__name__)
        else:
            em, f1 = score(answer.answer, items[index]["answer"])
            intended = start + due
            record.update(
                ok=True,
                e2e_ms=(end - intended) * 1000,
                lag_ms=(began - intended) * 1000,
                embed_rtt_ms=(middle - began) * 1000,
                qa_rtt_ms=(end - middle) * 1000,
                embed_queue_ms=found.queue_s * 1000,
                embed_service_ms=found.service_s * 1000,
                qa_queue_ms=answer.queue_s * 1000,
                qa_service_ms=answer.service_s * 1000,
                em=em,
                f1=f1,
                answer=answer.answer,
            )
        records[index] = record

    tasks = []
    start = time.perf_counter()
    for index, due in enumerate(arrivals):
        wait = start + due - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        tasks.append(asyncio.create_task(one(index, due)))

    if tasks:
        _, pending = await asyncio.wait(tasks, timeout=args.timeout + 5)
        for task in pending:
            task.cancel()

    summary = summarize(args, arrivals, records)
    out = args.out
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    with open(out.replace(".json", ".requests.jsonl"), "w") as f:
        for index in sorted(records):
            f.write(json.dumps(records[index]) + "\n")
    print(json.dumps(summary))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embed", required=True)
    p.add_argument("--qa", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--rate", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--warmup", type=float, default=5)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--search", default="single_pass")
    p.add_argument("--seed", type=int, default=8001)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--slo-ms", type=float, default=None)
    p.add_argument("--allow-empty", action="store_true")
    p.add_argument("--out", required=True)
    asyncio.run(run(p.parse_args()))


main()