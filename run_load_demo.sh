#!/usr/bin/env bash
# Sends open loop load to the deployed pipeline while the dashboard is open
cd "$(dirname "$0")/loadgen"
python3 - "${1:-8.0}" "${2:-120}" <<'PY'
import json, sys
from harness import RESULTS, probe
ref = json.loads((RESULTS / "sla.json").read_text())["reference"]
r = probe("demo_load", float(sys.argv[1]), int(sys.argv[2]), 0, ref["k"], ref["search"])
print("p50", round(r["p50_ms"]), "p95", round(r["p95_ms"]), "rps", round(r["achieved_rps"], 2))
PY
