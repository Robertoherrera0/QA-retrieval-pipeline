#!/bin/bash
set -e
cd "$(dirname "$0")"

command -v helm >/dev/null || curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
    -n monitoring --create-namespace -f monitoring-values.yaml --wait --timeout 10m

kubectl -n monitoring create configmap qa-dashboard --from-file=dashboard.json \
    --dry-run=client -o yaml | kubectl apply -f -
kubectl -n monitoring label configmap qa-dashboard grafana_dashboard=1 --overwrite