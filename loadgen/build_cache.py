import argparse
import asyncio
import json

import grpc
from google.protobuf import json_format

import qa_pb2
import qa_pb2_grpc

ROUND_ROBIN = json.dumps({"loadBalancingConfig": [{"round_robin": {}}]})


async def build(args):
    with open(args.pool, encoding="utf-8") as f:
        items = [json.loads(line) for _, line in zip(range(args.count), f)]

    channel = grpc.aio.insecure_channel(
        args.embed, options=[("grpc.service_config", ROUND_ROBIN)]
    )
    embed = qa_pb2_grpc.EmbeddingStub(channel)
    gate = asyncio.Semaphore(args.concurrency)
    chunks = [None] * len(items)

    async def one(index, item):
        request = qa_pb2.RetrieveRequest(
            question=item["question"],
            paragraphs=[
                qa_pb2.Paragraph(title=p["title"], para_id=n, text=p["text"])
                for n, p in enumerate(item["paragraphs"])
            ],
            k=args.k,
            search=args.search,
        )
        async with gate:
            for attempt in range(3):
                try:
                    reply = await embed.Retrieve(request, timeout=args.timeout)
                    chunks[index] = [
                        json_format.MessageToDict(c, preserving_proto_field_name=True)
                        for c in reply.chunks
                    ]
                    return
                except Exception:
                    await asyncio.sleep(2)
            raise RuntimeError(f"retrieval failed for {item['id']}")

    await asyncio.gather(*(one(i, item) for i, item in enumerate(items)))

    with open(args.out, "w", encoding="utf-8") as f:
        for item, found in zip(items, chunks):
            f.write(json.dumps({"id": item["id"], "chunks": found}) + "\n")
    print("cached", len(items), "questions")


p = argparse.ArgumentParser()
p.add_argument("--embed", required=True)
p.add_argument("--pool", required=True)
p.add_argument("--count", type=int, default=2000)
p.add_argument("--k", type=int, required=True)
p.add_argument("--search", default="single_pass")
p.add_argument("--concurrency", type=int, default=16)
p.add_argument("--timeout", type=float, default=120)
p.add_argument("--out", required=True)
asyncio.run(build(p.parse_args()))