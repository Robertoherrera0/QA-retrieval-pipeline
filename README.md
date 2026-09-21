# QA-Retrieval-Pipeline

Retrieval-augmented extractive question answering pipeline. A summarization step, implemented as retrieval over a ChromaDB vectorstore, reduces long contexts before passing them to a BERT-family extractive QA model, avoiding the 512 token limit these models impose.

## Pipeline

## Datasets


## Models
**Embedding models**
- sentence-transformers/all-MiniLM-L6-v2
- sentence-transformers/all-mpnet-base-v2
- sentence-transformers/all-MiniLM-L12-v2

**QA models**
- deepset/tinyroberta-squad2
- deepset/roberta-base-squad2
- google-bert/bert-large-uncased-whole-word-masking-finetuned-squad


## How to run
