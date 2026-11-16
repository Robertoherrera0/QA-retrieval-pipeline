import os

import grpc

import qa_pb2
import qa_pb2_grpc

EMBED_ADDR = os.environ.get("EMBED_ADDR", "localhost:50051")
QA_ADDR = os.environ.get("QA_ADDR", "localhost:50052")

paragraphs = [
    ("Eiffel Tower", "The Eiffel Tower is a wrought iron tower in Paris. It was finished in 1889."),
    ("Danube", "The Danube is a river that flows through Vienna and Budapest."),
    ("Mount Fuji", "Mount Fuji is the highest mountain in Japan. It is located on Honshu."),
]

question = "In which city is the Eiffel Tower?"

embed = qa_pb2_grpc.EmbeddingStub(grpc.insecure_channel(EMBED_ADDR))
qa = qa_pb2_grpc.QAStub(grpc.insecure_channel(QA_ADDR))

found = embed.Retrieve(qa_pb2.RetrieveRequest(
    question=question,
    paragraphs=[
        qa_pb2.Paragraph(title=title, para_id=i, text=text)
        for i, (title, text) in enumerate(paragraphs)
    ],
    k=3,
    search="single_pass",
))
answer = qa.Answer(qa_pb2.AnswerRequest(question=question, chunks=found.chunks))

print("chunks", [c.chunk_id for c in found.chunks])
print("answer", repr(answer.answer), "score", round(answer.score, 3))
print("embedding stage", {
    "chunking_s": found.chunking_s, "embedding_s": found.embedding_s,
    "indexing_s": found.indexing_s, "retrieval_s": found.retrieval_s,
})
print("QA model stage", {"packing_s": answer.packing_s, "qa_s": answer.qa_s})