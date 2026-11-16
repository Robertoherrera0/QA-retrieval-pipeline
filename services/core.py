from time import perf_counter

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_splitter(chunk_size, chunk_overlap, tokenizer, token_limit):
    # chunk_size counts the special tokens the embedding model adds
    if not 0 < chunk_size <= token_limit:
        raise ValueError("chunk_size is larger than the embedding model allows")

    n_special = tokenizer.num_special_tokens_to_add(pair=False)
    max_tokens = chunk_size - n_special  # room left for the text itself

    if not 0 <= chunk_overlap < max_tokens:
        raise ValueError("chunk_overlap must be smaller than the chunk size")

    def count_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))

    return RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        add_start_index=True,
    )


def retrieve(store, question, k, method, query_tokenizer,
             max_query_tokens, timings=None):
    def now():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return perf_counter()

    started = now()

    if k < 1:
        raise ValueError("k must be at least 1")
    if method not in ("single_pass", "two_hop"):
        raise ValueError("method must be 'single_pass' or 'two_hop'")
    if store is None:
        if timings is not None:
            timings["retrieval_s"] = now() - started
        return {"chunks": [], "expanded_query": None}

    def token_count(text):
        return len(query_tokenizer.encode(text, add_special_tokens=True))

    if token_count(question) > max_query_tokens:
        raise ValueError("question is longer than the embedding model allows")

    first = store.similarity_search_with_score(
        question, k=k if method == "single_pass" else 1
    )
    selected = [(doc, distance, 1) for doc, distance in first]
    expanded_query = None

    if method == "two_hop" and k > 1 and first:
        evidence = first[0][0].page_content
        offsets = query_tokenizer(
            evidence, add_special_tokens=False,
            return_offsets_mapping=True, truncation=False,
        )["offset_mapping"]

        end = min(len(offsets), max_query_tokens - token_count(question))
        while end > 0:
            candidate = question + "\n" + evidence[:offsets[end - 1][1]]
            if token_count(candidate) <= max_query_tokens:
                expanded_query = candidate
                break
            end -= 1

        if expanded_query is None:
            raise ValueError("no first hop evidence fits next to the question")

        skip_first_paragraph = {"para_id": {"$ne": first[0][0].metadata["para_id"]}}
        seen = {doc.metadata["chunk_id"] for doc, _, _ in selected}
        second = store.similarity_search_with_score(
            expanded_query, k=k, filter=skip_first_paragraph
        )

        for doc, distance in second:
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id not in seen:
                selected.append((doc, distance, 2))
                seen.add(chunk_id)
            if len(selected) == k:
                break

    result = {
        "chunks": [
            {"rank": rank, "text": doc.page_content, "metadata": dict(doc.metadata),
             "distance": float(distance), "hop": hop}
            for rank, (doc, distance, hop) in enumerate(selected, start=1)
        ],
        "expanded_query": expanded_query,
    }

    if timings is not None:
        timings["retrieval_s"] = now() - started

    return result


def pack_context(question, chunks, qa_tokenizer,
                 max_tokens=512, timings=None):
    started = perf_counter()

    def input_length(context):
        return len(qa_tokenizer(
            question, context, add_special_tokens=True,
            truncation=False,
        )["input_ids"])

    if input_length("") >= max_tokens:
        raise ValueError("The question leaves no room for QA context.")

    context = ""
    included = []

    for chunk in chunks:
        text = chunk["text"]
        separator = "\n\n" if context else ""
        candidate = context + separator + text
        truncated = False

        if input_length(candidate) > max_tokens:
            offsets = qa_tokenizer(
                text, add_special_tokens=False,
                return_offsets_mapping=True, truncation=False,
            )["offset_mapping"]
            end = min(len(offsets), max_tokens)

            while end > 0:
                prefix = text[:offsets[end - 1][1]]
                candidate = context + separator + prefix
                if input_length(candidate) <= max_tokens:
                    text = prefix
                    break
                end -= 1

            if end == 0:
                break
            truncated = True

        context_start = len(context) + len(separator)
        context = candidate
        metadata = dict(chunk["metadata"])
        metadata["end_index"] = metadata["start_index"] + len(text)

        included.append({
            "rank": chunk["rank"],
            "text": text,
            "metadata": metadata,
            "context_start": context_start,
            "context_end": len(context),
            "truncated": truncated,
        })

        if truncated:
            break

    result = {
        "context": context,
        "input_tokens": input_length(context),
        "included_chunks": included,
    }

    if timings is not None:
        timings["packing_s"] = perf_counter() - started

    return result