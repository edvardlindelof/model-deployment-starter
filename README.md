# Machine learning model deployment starter
A starter project for developing, training and deploying prediction models.

Key technologies involved are MLflow, Hugging Face Transformers, DVC and Jupyter.
With services containerized and storage delegated to S3 and SQL, the solution is readily transferable to the cloud.

## Requirements

### uv
On Ubuntu:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### direnv
On Ubuntu:
```bash
sudo apt install direnv
```

Then add direnv to your shell. For bash:
```bash
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
```

### Docker
[Get Docker](https://docs.docker.com/get-started/get-docker/)

### docker-compose
On Ubuntu:
```bash
sudo apt install docker.io docker-compose
```

## Run
```bash
docker-compose up
```

## Test
```bash
uv run pytest
```
