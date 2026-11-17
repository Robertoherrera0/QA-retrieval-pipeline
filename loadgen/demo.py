import argparse
import json

import grpc

import qa_pb2
import qa_pb2_grpc
from loadgen import score

p = argparse.ArgumentParser()
p.add_argument("--embed", required=True)
p.add_argument("--qa", required=True)
p.add_argument("--pool", default="/data/testslice.jsonl")
p.add_argument("--index", type=int, default=0)
p.add_argument("--k", type=int, default=5)
p.add_argument("--search", default="single_pass")
p.add_argument("--gold", action="store_true", help="answer from the gold paragraphs only")
args = p.parse_args()

with open(args.pool, encoding="utf-8") as f:
    item = json.loads(f.read().splitlines()[args.index])

embed = qa_pb2_grpc.EmbeddingStub(grpc.insecure_channel(args.embed))
qa = qa_pb2_grpc.QAStub(grpc.insecure_channel(args.qa))

print("Question  ", item["question"])
print("Paragraphs", len(item["paragraphs"]), "sent with the request")

if args.gold:
    chunks = [
        qa_pb2.Chunk(rank=n, text=q["text"], para_id=i, title=q["title"],
                     start_index=0, end_index=len(q["text"]), chunk_id=f"gold_{i}", hop=1)
        for n, (i, q) in enumerate(
            (i, q) for i, q in enumerate(item["paragraphs"]) if q["title"] in item["gold_titles"]
        )
    ]
    print("Gold input", len(chunks), "supporting paragraphs, no retrieval")
else:
    found = embed.Retrieve(qa_pb2.RetrieveRequest(
        question=item["question"],
        paragraphs=[qa_pb2.Paragraph(title=q["title"], para_id=i, text=q["text"])
                    for i, q in enumerate(item["paragraphs"])],
        k=args.k, search=args.search,
    ), timeout=120)
    chunks = found.chunks
    print()
    print("Retrieved chunks")
    for c in chunks:
        gold = "gold" if c.title in item["gold_titles"] else "    "
        print(f"  {c.rank} {gold} {c.title[:28]:28} {c.text[:70]!r}")
    print()
    print(f"Embedding stage  chunking {found.chunking_s*1000:.0f} ms  embedding {found.embedding_s*1000:.0f} ms  "
          f"indexing {found.indexing_s*1000:.0f} ms  retrieval {found.retrieval_s*1000:.0f} ms  "
          f"queue {found.queue_s*1000:.0f} ms  service {found.service_s*1000:.0f} ms")

reply = qa.Answer(qa_pb2.AnswerRequest(question=item["question"], chunks=chunks), timeout=120)
em, f1 = score(reply.answer, item["answer"])
print(f"QA stage         packing {reply.packing_s*1000:.0f} ms  model {reply.qa_s*1000:.0f} ms  "
      f"queue {reply.queue_s*1000:.0f} ms  service {reply.service_s*1000:.0f} ms")
print()
print("Answer    ", reply.answer)
print("Gold      ", item["answer"])
print(f"Score      F1 {f1:.2f}  EM {em:.0f}  confidence {reply.score:.2f}")