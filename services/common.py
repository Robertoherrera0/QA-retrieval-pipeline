import os
import queue
import threading
from concurrent import futures
from concurrent.futures import Future
from pathlib import Path
from time import perf_counter

import grpc
from minio import Minio
from prometheus_client import Counter, Gauge, Histogram, start_http_server

BUCKET = "models"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))

REQUESTS = Counter("requests_total", "Finished requests", ["stage", "status"])
LATENCY = Histogram(
    "request_seconds", "Seconds from arrival to reply", ["stage"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
QUEUE_WAIT = Histogram(
    "queue_wait_seconds", "Seconds a request waited before its batch started",
    ["stage"], buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5),
)
BATCH = Histogram(
    "batch_size", "Requests per batch", ["stage"], buckets=(1, 2, 4, 8, 16, 32),
)
QUEUE_DEPTH = Gauge("queue_depth", "Requests waiting", ["stage"])
LOAD = Gauge("model_load_seconds", "Seconds to fetch and load the model", ["stage"])


def set_threads():
    threads = os.environ.get("NUM_THREADS")
    if threads:
        import torch
        torch.set_num_threads(int(threads))


def fetch_model(stage, name):
    client = Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ.get("MINIO_USER", "qa_registry"),
        secret_key=os.environ.get("MINIO_PASSWORD", "qa_registry_pass"),
        secure=False,
    )
    started = perf_counter()
    objects = list(client.list_objects(BUCKET, prefix=f"{stage}/{name}/", recursive=True))
    versions = sorted({o.object_name.split("/")[2] for o in objects})
    version = os.environ.get("MODEL_VERSION")
    if version is None:
        if len(versions) != 1:
            raise RuntimeError(f"{stage}/{name} has versions {versions}, set MODEL_VERSION")
        version = versions[0]

    prefix = f"{stage}/{name}/{version}/"
    folder = MODEL_DIR / stage / name / version
    found = False
    for obj in objects:
        if not obj.object_name.startswith(prefix):
            continue
        found = True
        if obj.object_name.endswith("metadata.json"):
            continue
        target = folder / obj.object_name[len(prefix):]
        target.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(BUCKET, obj.object_name, str(target))
    if not found:
        raise RuntimeError(f"{prefix} is not in the registry")
    return folder, version, perf_counter() - started


class Batcher:
    def __init__(self, stage, run_batch):
        self.stage = stage
        self.run_batch = run_batch
        self.batch_size = int(os.environ.get("BATCH_SIZE", "1"))
        self.timeout = float(os.environ.get("BATCH_TIMEOUT_MS", "10")) / 1000
        self.queue = queue.Queue(maxsize=int(os.environ.get("MAX_QUEUE", "1000")))
        QUEUE_DEPTH.labels(stage).set_function(self.queue.qsize)
        threading.Thread(target=self.loop, daemon=True).start()

    def call(self, item):
        arrived = perf_counter()
        future = Future()
        self.queue.put((item, future, arrived))
        try:
            result = future.result()
        except Exception:
            REQUESTS.labels(self.stage, "error").inc()
            raise
        REQUESTS.labels(self.stage, "ok").inc()
        LATENCY.labels(self.stage).observe(perf_counter() - arrived)
        return result

    def loop(self):
        while True:
            batch = [self.queue.get()]
            deadline = perf_counter() + self.timeout
            while len(batch) < self.batch_size:
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break

            started = perf_counter()
            for _, _, arrived in batch:
                QUEUE_WAIT.labels(self.stage).observe(started - arrived)
            BATCH.labels(self.stage).observe(len(batch))

            try:
                results = self.run_batch([item for item, _, _ in batch])
            except Exception as error:
                results = [error] * len(batch)
            for (_, future, _), result in zip(batch, results):
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)


def call_or_abort(batcher, request, context):
    try:
        return batcher.call(request)
    except Exception as error:
        context.abort(grpc.StatusCode.INTERNAL, str(error))


def serve(register, servicer):
    start_http_server(int(os.environ.get("METRICS_PORT", "8000")))
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "32")))
    )
    register(servicer, server)
    server.add_insecure_port("[::]:" + os.environ.get("GRPC_PORT", "50051"))
    server.start()
    server.wait_for_termination()