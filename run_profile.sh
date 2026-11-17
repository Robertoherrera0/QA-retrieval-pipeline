#!/bin/bash
# usage run_profile.sh [report|t7|full]
#   report  builds every table and plot from the saved results (default, takes seconds)
#   t7      runs the end to end configurations and the gold input ablation, then builds the report
#   full    deploys, calibrates the SLA, runs the whole profiling sweep, runs t7, then builds the report
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

end_to_end() {
    python3 loadgen/e2e.py accuracy
    python3 loadgen/e2e.py load
    python3 loadgen/e2e.py gold
}

case "$mode" in
    report)
        report
        ;;
    t7)
        end_to_end
        report
        ;;
    full)
        bash k8s/deploy.sh
        python3 loadgen/calibrate.py --embedder "$EMBEDDER" --qa-model "$QA_MODEL" \
            --k "$K" --search "$SEARCH"
        python3 loadgen/sweep.py
        end_to_end
        report
        ;;
    *)
        echo "usage run_profile.sh [report|t7|full]"
        exit 1
        ;;
esac