#!/bin/bash
set -e
cd "$(dirname "$0")"

host=$(kubectl get node -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

kubectl create namespace qa --dry-run=client -o yaml | kubectl apply -f -
kubectl -n qa create configmap qa-config --from-literal=MINIO_ENDPOINT="$host:9000" \
    --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f deployments.yaml
kubectl apply -f servicemonitors.yaml

for stage in embed qa; do
    kubectl -n qa rollout status deployment/$stage --timeout=600s
done