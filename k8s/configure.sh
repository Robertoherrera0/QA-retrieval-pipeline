#!/bin/bash
# usage configure.sh embed|qa REPLICAS CORES [NAME=value ...]
set -e
stage=$1
replicas=$2
cores=$3
shift 3

kubectl -n qa rollout pause deployment/$stage
kubectl -n qa set resources deployment/$stage \
    --requests=cpu=$cores,memory=2Gi --limits=cpu=$cores,memory=6Gi
kubectl -n qa set env deployment/$stage NUM_THREADS=$cores "$@"
kubectl -n qa scale deployment/$stage --replicas=$replicas
kubectl -n qa rollout resume deployment/$stage
kubectl -n qa rollout status deployment/$stage --timeout=600s