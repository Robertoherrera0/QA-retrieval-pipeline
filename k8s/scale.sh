#!/bin/bash
# usage scale.sh embed|qa REPLICAS
set -e
kubectl -n qa scale deployment/$1 --replicas=$2
kubectl -n qa rollout status deployment/$1 --timeout=600s