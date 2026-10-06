## Run the notebooks

From the cloned repository folder, build the image:

```bash
docker build -t qa-retrieval-pipeline .
```

Start the container with the project folder mounted at `/notebooks` and a persistent download cache.

```bash
docker run --name qa -p 8888:8888 -v "/$(pwd):/notebooks" -v qa_cache:/root/.cache qa-retrieval-pipeline
```
