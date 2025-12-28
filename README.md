# Machine learning model deployment starter
A starter project for developing, training and deploying prediction models.

Key technologies involved are MLflow, Hugging Face Transformers, DVC and Jupyter.
With services containerized and storage delegated to S3 and SQL, the solution is readily transferable to the cloud.

## Requirements
These are the tool installation steps for Ubuntu.
- uv:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- direnv:
  ```bash
  sudo apt install direnv
  echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
  ```
- Docker: [Get Docker](https://docs.docker.com/get-started/get-docker/)
- docker compose:
  ```bash
  sudo apt install docker-compose-plugin
  ```
- DVC:
  ```bash
  uv tool install dvc[s3]
  ```

## Run
```bash
cp .env.local .env
direnv allow
```

```bash
docker compose --profile tracking up -d
docker compose --profile serving up -d
```

## Version controlling data
- install with `dvc install`
- store data in data/
- track added and changed files with `dvc add data/<the file>`
- pull/push changes from/to the MinIO service with `dvc pull` and `dvc push`
- the copy of the files in data/ are automatically kept in sync with the checked out git commit through pre-commit hooks
