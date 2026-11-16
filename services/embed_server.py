import os
from time import perf_counter
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

import qa_pb2
import qa_pb2_grpc
from common import LOAD, Batcher, call_or_abort, fetch_model, serve, set_threads
from core import make_splitter, retrieve

STAGE = "embedding"
NAME = os.environ["EMBEDDER"]
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "128"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "0"))

set_threads()
path, version, fetch_s = fetch_model(STAGE, NAME)
started = perf_counter()
embedder = HuggingFaceEmbeddings(model_name=str(path), model_kwargs={"device": "cpu"})
LOAD.labels(STAGE).set(fetch_s + perf_counter() - started)

tokenizer = embedder._client.tokenizer
max_len = embedder._client.max_seq_length
splitter = make_splitter(CHUNK_SIZE, CHUNK_OVERLAP, tokenizer, max_len)


class Fixed(Embeddings):
    # serves vectors that were already computed for the whole batch
    def __init__(self, vectors, base):
        self.vectors = vectors
        self.base = base

    def embed_documents(self, texts):
        return [self.vectors[t] for t in texts]

    def embed_query(self, text):
        return self.base.embed_query(text)


def chunk(paragraphs):
    docs = []
    for p in paragraphs:
        metadata = {"title": p.title, "para_id": p.para_id}
        for piece in splitter.create_documents([p.text], metadatas=[metadata]):
            start = piece.metadata["start_index"]
            end = start + len(piece.page_content)
            if start < 0 or p.text[start:end] != piece.page_content:
                raise ValueError("chunk does not match the paragraph text")
            n_tokens = len(tokenizer.encode(
                piece.page_content, add_special_tokens=True, truncation=False
            ))
            if n_tokens > CHUNK_SIZE:
                raise ValueError("chunk is longer than chunk_size")
            piece.metadata.update({
                "end_index": end,
                "chunk_id": f"{p.para_id}:{start}:{end}",
                "embedding_tokens": n_tokens,
            })
            docs.append(piece)
    return docs


def run_batch(requests):
    chunked = []
    for r in requests:
        began = perf_counter()
        try:
            chunked.append((chunk(r.paragraphs), perf_counter() - began))
        except Exception as error:
            chunked.append((error, 0.0))

    texts = sorted({
        d.page_content for docs, _ in chunked
        if not isinstance(docs, Exception) for d in docs
    })
    began = perf_counter()
    vectors = embedder.embed_documents(texts) if texts else []
    embed_s = perf_counter() - began
    fixed = Fixed(dict(zip(texts, vectors)), embedder)

    results = []
    for r, (docs, chunk_s) in zip(requests, chunked):
        if isinstance(docs, Exception):
            results.append(docs)
            continue
        try:
            began = perf_counter()
            store = None
            if docs:
                store = Chroma.from_documents(
                    documents=docs,
                    embedding=fixed,
                    ids=[d.metadata["chunk_id"] for d in docs],
                    collection_name="retrieval_" + uuid4().hex,
                    collection_metadata={"hnsw:space": "cosine"},
                )
            index_s = perf_counter() - began

            timings = {}
            found = retrieve(store, r.question, r.k, r.search, tokenizer, max_len, timings)
            if store is not None:
                store.delete_collection()

            chunks = []
            for c in found["chunks"]:
                m = c["metadata"]
                chunks.append(qa_pb2.Chunk(
                    rank=c["rank"], text=c["text"], para_id=m["para_id"],
                    title=m["title"], start_index=m["start_index"],
                    end_index=m["end_index"], chunk_id=m["chunk_id"],
                    distance=c["distance"], hop=c["hop"],
                ))
            results.append(qa_pb2.RetrieveReply(
                chunks=chunks, chunking_s=chunk_s, embedding_s=embed_s,
                indexing_s=index_s, retrieval_s=timings["retrieval_s"],
                batch_size=len(requests),
            ))
        except Exception as error:
            results.append(error)
    return results


batcher = Batcher(STAGE, run_batch)


class Embedding(qa_pb2_grpc.EmbeddingServicer):
    def Retrieve(self, request, context):
        return call_or_abort(batcher, request, context)


serve(qa_pb2_grpc.add_EmbeddingServicer_to_server, Embedding())