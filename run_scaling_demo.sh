#!/usr/bin/env bash
cd "$(dirname "$0")"
RATE="${1:-3.0}"
kubectl -n qa scale deploy/embed --replicas=1
kubectl -n qa scale deploy/qa --replicas=1
kubectl -n qa rollout status deploy/embed
kubectl -n qa rollout status deploy/qa
bash run_load_demo.sh "$RATE" 90
kubectl -n qa scale deploy/embed --replicas=4
kubectl -n qa scale deploy/qa --replicas=4
kubectl -n qa rollout status deploy/embed
kubectl -n qa rollout status deploy/qa
sleep 30
bash run_load_demo.sh "$RATE" 90
