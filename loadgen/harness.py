import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_HOST", REPO / "results" / "chameleonTACC"))
POOL = RESULTS / "loadpool.jsonl"


def target(stage):
    data = json.loads(subprocess.check_output(
        ["kubectl", "-n", "qa", "get", "endpoints", stage, "-o", "json"]
    ))
    addresses = [
        f"{a['ip']}:50051"
        for subset in data.get("subsets") or []
        for a in subset.get("addresses") or []
    ]
    if not addresses:
        raise RuntimeError(f"no ready pods for {stage}")
    return "ipv4:" + ",".join(addresses)


def configure(stage, replicas, cores, **env):
    settings = [f"{name}={value}" for name, value in env.items()]
    subprocess.run(
        ["bash", str(REPO / "k8s" / "configure.sh"), stage, str(replicas), str(cores)]
        + settings,
        check=True,
    )


CACHE = RESULTS / "chunks_cache.jsonl"


def build_cache(k, search, count=2000):
    subprocess.run(
        [
            "docker", "run", "--rm", "--network", "host", "-v", f"{RESULTS}:/data",
            "qa-loadgen", "python", "build_cache.py",
            "--embed", target("embed"), "--pool", "/data/loadpool.jsonl",
            "--count", str(count), "--k", str(k), "--search", search,
            "--out", f"/data/{CACHE.name}",
        ],
        check=True,
    )


def probe(tag, rate, duration, offset, k, search, slo_ms=None, warmup=5, cached=False):
    out_dir = RESULTS / "loadruns"
    out_dir.mkdir(parents=True, exist_ok=True)
    upstream = ["--cache", f"/data/{CACHE.name}"] if cached else ["--embed", target("embed")]
    command = [
        "docker", "run", "--rm", "--network", "host", "-v", f"{RESULTS}:/data",
        "qa-loadgen", "python", "loadgen.py",
        *upstream, "--qa", target("qa"),
        "--pool", "/data/loadpool.jsonl", "--rate", str(rate),
        "--duration", str(duration), "--warmup", str(warmup),
        "--offset", str(offset), "--k", str(k), "--search", search,
        "--out", f"/data/loadruns/{tag}.json",
    ]
    if slo_ms:
        command += ["--slo-ms", str(slo_ms)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return json.loads((out_dir / f"{tag}.json").read_text())