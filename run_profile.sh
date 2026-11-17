#!/bin/bash
# usage run_profile.sh [report|full]
#   report  builds every table and plot from the saved results (default, takes seconds)
#   full    deploys, calibrates the SLA, runs the whole profiling sweep, then builds the report
set -e
cd "$(dirname "$0")"

mode=${1:-report}
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/config}
results=results/chameleonTACC

EMBEDDER=all-MiniLM-L6-v2
QA_MODEL=tinyroberta-squad2
K=5
SEARCH=single_pass

report() {
    if python3 -c "import numpy, matplotlib" 2>/dev/null; then
        python3 loadgen/report.py "$results"
    else
        docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
            -v "$PWD":/work -w /work qa-pipeline loadgen/report.py "$results"
    fi
}

case "$mode" in
    report)
        report
        ;;
    full)
        bash k8s/deploy.sh
        python3 loadgen/calibrate.py --embedder "$EMBEDDER" --qa-model "$QA_MODEL" \
            --k "$K" --search "$SEARCH"
        python3 loadgen/sweep.py
        report
        ;;
    *)
        echo "usage run_profile.sh [report|full]"
        exit 1
        ;;
esac