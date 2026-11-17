import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results/chameleonTACC")
OUT = RESULTS / "report"
OUT.mkdir(exist_ok=True)

FILE_PATTERN = re.compile(r"^(embed|qa)_(.+)_(\d+)x(\d+)_b(\d+)_[\d.]+\.requests\.jsonl$")


def load_rows():
    latest = {}
    for line in (RESULTS / "profiling.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["stage"], r["variant"], r["replicas"], r["cores"], r["batch"])] = r
    return list(latest.values())


def accuracy_by_variant():
    answers = defaultdict(dict)
    for path in sorted((RESULTS / "loadruns").glob("*.requests.jsonl")):
        m = FILE_PATTERN.match(path.name)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["ok"] and r["question_id"] not in answers[key]:
                answers[key][r["question_id"]] = (r["f1"], r["em"])
    result = {}
    for stage in ("embed", "qa"):
        keys = [k for k in answers if k[0] == stage]
        if not keys:
            continue
        common = sorted(set.intersection(*(set(answers[k]) for k in keys)))
        for k in keys:
            f1 = [answers[k][q][0] for q in common]
            em = [answers[k][q][1] for q in common]
            result[k] = {"f1": float(np.mean(f1)), "em": float(np.mean(em)),
                         "questions": len(common)}
    return result


def interpolated_max(row):
    slo = row["slo_ms"]
    probes = row["probes"]
    passing = [p for p in probes
               if p["p95_ms"] <= slo and p["failed"] <= 0.01 * p["sent"]]
    if not passing:
        return None
    low = max(passing, key=lambda p: p["rate"])
    failing = [p for p in probes if p["rate"] > low["rate"] and p not in passing]
    if not failing:
        return None
    high = min(failing, key=lambda p: p["rate"])
    if high["p95_ms"] <= low["p95_ms"]:
        return None
    share = (slo - low["p95_ms"]) / (high["p95_ms"] - low["p95_ms"])
    return low["rate"] + share * (high["rate"] - low["rate"])


def table_rows(rows, accuracy):
    table = []
    for r in sorted(rows, key=lambda r: (r["stage"], r["variant"], r["cost"],
                                         r["replicas"], r["batch"])):
        a = r["at_max"]
        acc = accuracy.get((r["stage"], r["variant"]))
        row = {
            "stage": r["stage"],
            "variant": r["variant"],
            "batch": r["batch"],
            "replicas": r["replicas"],
            "cores_per_replica": r["cores"],
            "cost": r["cost"],
            "p50_ms": None, "p95_ms": None, "p99_ms": None,
            "max_rps_meeting_slo": r["max_rps_meeting_slo"] or None,
            "max_rps_interpolated": interpolated_max(r),
            "accuracy_f1": acc["f1"] if acc else None,
            "accuracy_em": acc["em"] if acc else None,
            "accuracy_questions": acc["questions"] if acc else None,
            "queue_ms": None, "inference_ms": None, "network_ms": None,
            "other_stage_queue_ms": None,
            "notes": "",
        }
        if a:
            own = "embed" if r["stage"] == "embed" else "qa"
            other = "qa" if own == "embed" else "embed"
            row.update(
                p50_ms=a["p50_ms"], p95_ms=a["p95_ms"], p99_ms=a["p99_ms"],
                queue_ms=a[f"{own}_queue_ms"],
                inference_ms=a[f"{own}_service_ms"],
                network_ms=a[f"{own}_network_ms"],
                other_stage_queue_ms=a[f"{other}_queue_ms"],
            )
        else:
            lowest = min(p["rate"] for p in r["probes"])
            row["notes"] = f"no rate at or above {lowest:.2f} met the SLO"
        table.append(row)
    return table


def fmt(value, digits=0):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}" if digits else f"{round(value)}"


def write_table(table):
    columns = list(table[0].keys())
    with (OUT / "t6_profiling.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)

    heads = ["Stage", "Variant", "Batch", "Replicas", "Cores/replica", "Cost",
             "p50 ms", "p95 ms", "p99 ms", "Max throughput meeting SLO (req/s)",
             "Accuracy (F1)", "Queue ms", "Inference ms", "Network ms", "Notes"]
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for r in table:
        cells = [
            r["stage"], r["variant"], str(r["batch"]), str(r["replicas"]),
            str(r["cores_per_replica"]), str(r["cost"]),
            fmt(r["p50_ms"]), fmt(r["p95_ms"]), fmt(r["p99_ms"]),
            fmt(r["max_rps_meeting_slo"], 2), fmt(r["accuracy_f1"], 3),
            fmt(r["queue_ms"]), fmt(r["inference_ms"]), fmt(r["network_ms"]),
            r["notes"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    (OUT / "t6_profiling.md").write_text("\n".join(lines) + "\n")


def scatter(table, xkey, xlabel, filename, log=False):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    for ax, stage in zip(axes, ("embed", "qa")):
        subset = [r for r in table if r["stage"] == stage
                  and r[xkey] is not None and r["accuracy_f1"] is not None]
        variants = sorted({r["variant"] for r in subset})
        for index, variant in enumerate(variants):
            pts = [r for r in subset if r["variant"] == variant]
            ax.scatter([r[xkey] for r in pts], [r["accuracy_f1"] for r in pts],
                       s=[20 + 12 * r["cost"] for r in pts],
                       color=f"C{index}", alpha=0.75, label=variant)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (F1)")
        ax.set_title("Embedding stage" if stage == "embed" else "QA stage")
        ax.grid(alpha=0.25)
        if variants:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / filename, facecolor="white")
    plt.close(fig)


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pareto_front(points):
    front = []
    best = -1.0
    for point in sorted(points, key=lambda p: (p["cost"], -p["f1"])):
        if point["f1"] > best:
            front.append(point)
            best = point["f1"]
    return front


def t7():
    rows = read_lines(RESULTS / "e2e.jsonl")
    accuracy = {(r["embedder"], r["qa_model"]): r for r in rows if r["part"] == "accuracy"}
    configs = []
    for r in rows:
        if r["part"] != "load":
            continue
        a = accuracy.get((r["embedder"], r["qa_model"]))
        configs.append({
            "embedder": r["embedder"], "qa_model": r["qa_model"],
            "embed": f"{r['embed_replicas']}x{r['embed_cores']}",
            "qa": f"{r['qa_replicas']}x{r['qa_cores']}",
            "cost": r["cost"], "core_seconds": r["core_seconds"],
            "f1": a["f1"] if a else None, "em": a["em"] if a else None,
            "p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"], "p99_ms": r["p99_ms"],
            "slo_attainment": r["slo_attainment"], "meets_sla": r["meets_sla"],
            "target_reachable_in_profile": r["profile_says_target_reachable"],
        })
    if not configs:
        return
    with (OUT / "t7_end_to_end.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(configs[0].keys()))
        writer.writeheader()
        writer.writerows(configs)
    heads = ["Embedder", "QA model", "Embed replicas x cores", "QA replicas x cores", "Cost",
             "Core seconds", "F1", "EM", "p50 ms", "p95 ms", "p99 ms", "SLO attainment", "Meets SLA"]
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for c in configs:
        attainment = "" if c["slo_attainment"] is None else f"{100 * c['slo_attainment']:.1f}%"
        lines.append("| " + " | ".join([
            c["embedder"], c["qa_model"], c["embed"], c["qa"], str(c["cost"]),
            fmt(c["core_seconds"]), fmt(c["f1"], 3), fmt(c["em"], 3), fmt(c["p50_ms"]),
            fmt(c["p95_ms"]), fmt(c["p99_ms"]), attainment, "yes" if c["meets_sla"] else "no",
        ]) + " |")
    (OUT / "t7_end_to_end.md").write_text("\n".join(lines) + "\n")

    scored = [c for c in configs if c["f1"] is not None]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for c in scored:
        ax.scatter(c["cost"], c["f1"], s=70, marker="o" if c["meets_sla"] else "x",
                   color="C2" if c["meets_sla"] else "C3")
    front = pareto_front(scored)
    ax.step([c["cost"] for c in front], [c["f1"] for c in front], where="post",
            color="black", linewidth=1.2)
    for c in front:
        ax.annotate(f"{c['embedder']} + {c['qa_model'].split('-')[0]}", (c["cost"], c["f1"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.scatter([], [], marker="o", color="C2", label="meets the SLA")
    ax.scatter([], [], marker="x", color="C3", label="misses the SLA")
    ax.plot([], [], color="black", label="Pareto front")
    ax.set_xlabel("Cost (replicas x cores per replica, both stages)")
    ax.set_ylabel("Test slice F1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pareto_cost_accuracy.png", facecolor="white")
    plt.close(fig)

    gold = read_lines(RESULTS / "gold.jsonl")
    if gold:
        sla = json.loads((RESULTS / "sla.json").read_text())
        reference = sla["reference"]["embedder"]
        out = ["| QA model | F1 on gold paragraphs | F1 in the full pipeline | Gap |", "|---|---|---|---|"]
        for g in gold:
            full = accuracy.get((reference, g["qa_model"]))
            full_f1 = full["f1"] if full else None
            gap = None if full_f1 is None else g["f1"] - full_f1
            out.append(f"| {g['qa_model']} | {fmt(g['f1'], 3)} | {fmt(full_f1, 3)} | {fmt(gap, 3)} |")
        (OUT / "gold_ablation.md").write_text("\n".join(out) + "\n")
    print(len(configs), "end to end rows written to", OUT)


def main():
    rows = load_rows()
    accuracy = accuracy_by_variant()
    table = table_rows(rows, accuracy)
    write_table(table)
    scatter(table, "p95_ms", "p95 latency at the maximum rate meeting the SLO (ms)",
            "accuracy_vs_p95.png")
    scatter(table, "max_rps_meeting_slo", "Maximum throughput meeting the SLO (requests per second)",
            "accuracy_vs_throughput.png", log=True)
    (OUT / "accuracy.json").write_text(json.dumps(
        {f"{k[0]}/{k[1]}": v for k, v in accuracy.items()}, indent=2))
    print(len(table), "rows written to", OUT)
    t7()


main()