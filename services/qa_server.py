import os
from time import perf_counter

from transformers import pipeline

import qa_pb2
import qa_pb2_grpc
from common import LOAD, Batcher, call_or_abort, fetch_model, serve, set_threads
from core import pack_context

STAGE = "qa"
NAME = os.environ["QA_MODEL"]
QA_TOKEN_LIMIT = 512
MAX_ANSWER_LEN = 512

set_threads()
path, version, fetch_s = fetch_model(STAGE, NAME)
started = perf_counter()
qa_model = pipeline(
    "question-answering", model=str(path), tokenizer=str(path), device=-1
)
qa_model.model.eval()
LOAD.labels(STAGE).set(fetch_s + perf_counter() - started)


def run_batch(requests):
    packed = []
    for r in requests:
        began = perf_counter()
        try:
            chunks = [
                {"rank": c.rank, "text": c.text, "metadata": {"start_index": c.start_index}}
                for c in r.chunks
            ]
            p = pack_context(r.question, chunks, qa_model.tokenizer, QA_TOKEN_LIMIT)
            packed.append((p, perf_counter() - began))
        except Exception as error:
            packed.append((error, 0.0))

    predictions = {}
    qa_s = 0.0
    for allow_empty in (False, True):
        index = [
            i for i, (p, _) in enumerate(packed)
            if not isinstance(p, Exception) and p["context"].strip()
            and requests[i].allow_empty == allow_empty
        ]
        if not index:
            continue
        began = perf_counter()
        out = qa_model(
            question=[requests[i].question for i in index],
            context=[packed[i][0]["context"] for i in index],
            max_seq_len=QA_TOKEN_LIMIT,
            max_question_len=QA_TOKEN_LIMIT,
            max_answer_len=MAX_ANSWER_LEN,
            doc_stride=0,
            handle_impossible_answer=allow_empty,
            batch_size=len(index),
        )
        qa_s += perf_counter() - began
        if isinstance(out, dict):
            out = [out]
        predictions.update(zip(index, out))

    results = []
    for i, (p, pack_s) in enumerate(packed):
        if isinstance(p, Exception):
            results.append(p)
            continue
        prediction = predictions.get(i, {"answer": "", "score": 0.0})
        results.append(qa_pb2.AnswerReply(
            answer=prediction["answer"], score=float(prediction["score"]),
            input_tokens=p["input_tokens"], packing_s=pack_s, qa_s=qa_s,
            batch_size=len(requests),
        ))
    return results


batcher = Batcher(STAGE, run_batch)


class QA(qa_pb2_grpc.QAServicer):
    def Answer(self, request, context):
        return call_or_abort(batcher, request, context)


serve(qa_pb2_grpc.add_QAServicer_to_server, QA())