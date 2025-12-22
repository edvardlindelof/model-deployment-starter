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
- docker-compose:
  ```bash
  sudo apt install docker.io docker-compose
  ```
- DVC:
  ```bash
  uv tool install dvc
  ```

## Run
```bash
docker-compose up
```
