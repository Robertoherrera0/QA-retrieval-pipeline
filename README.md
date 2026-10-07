# QA-Pipeline

Retrieval-augmented extractive question answering pipeline. A summarization step, implemented as retrieval over a ChromaDB vectorstore, reduces long contexts before passing them to a BERT-family extractive QA model, avoiding the 512 token limit these models impose.

## Pipeline

Stage 1 splits each context into chunks, embeds the chunks with a sentence transformer, stores them in ChromaDB and retrieves the chunks nearest to the question. Stage 2 packs the question and the retrieved chunks into one 512 token window, and the QA model returns an answer span with a confidence score. An assertion passes when the confidence is at least 0.5. For every configuration the notebook reports exact match, F1, assertions passed, evidence recall and precision, and latency per stage.

Search methods are single pass and two hop. Chunking methods are recursive and semantic, with chunk sizes of 64, 128 and 256 tokens.

## Datasets

HotpotQA (Wikipedia paragraphs, multi hop questions) and QASPER (questions about full research papers). The notebook runs on 100 HotpotQA questions and 57 QASPER questions, drawn with a fixed seed.

## Models

Embedding models are all-MiniLM-L6-v2, all-MiniLM-L12-v2 and all-mpnet-base-v2. QA models are tinyroberta-squad2, roberta-base-squad2 and bert-large-uncased-whole-word-masking-finetuned-squad.

## How to run

Everything is in `qa_pipeline.ipynb`.

    docker build -t qa-pipeline .
    docker run -p 8888:8888 -v "$(pwd)/results:/app/results" qa-pipeline

Open the notebook in Jupyter and run all cells. The last cell runs all 57 settings per dataset and saves each run in `results/`. By default it uses a small sample of questions and takes about 15 minutes. To reproduce the numbers in the report on all 157 questions, set `QUICK = False` in that cell. This takes about 2 hours on a CPU laptop.

The reported results are in `results/laptopRH/final_results/combined_summary.csv`. The report is in `Project-Milestone-1.pdf`.
## Team

Roberto Herrera, Bach Nguyen
