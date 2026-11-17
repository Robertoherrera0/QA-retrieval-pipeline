#!/bin/bash
# usage bash run_demo.sh [QUESTION_INDEX]
set -e
cd "$(dirname "$0")"
export KUBECONFIG=${KUBECONFIG:-$HOME/.kube/config}
index=${1:-0}
results=$PWD/results/chameleonTACC

embed=$(kubectl -n qa get endpoints embed -o jsonpath='{.subsets[0].addresses[0].ip}'):50051
qa=$(kubectl -n qa get endpoints qa -o jsonpath='{.subsets[0].addresses[0].ip}'):50051

run_demo() {
    docker run --rm --network host \
        -v "$PWD/loadgen/demo.py":/app/demo.py -v "$results":/data \
        qa-loadgen python demo.py --embed "$embed" --qa "$qa" --index "$index" "$@"
}

echo "Services"
kubectl -n qa get pods
echo
echo "Full pipeline"
run_demo
echo
echo "Gold input ablation"
run_demo --gold