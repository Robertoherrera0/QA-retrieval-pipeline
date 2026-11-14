## Run the notebooks

From the cloned repository folder, build the image.

```bash
docker build -t qa-pipeline .
```

Start the container with the project folder mounted at `/work` and the results folder mounted at `/app/results`.

```bash
docker run -p 8888:8888 -v "$(pwd)/results:/app/results" -v "$(pwd):/work" qa-pipeline
```