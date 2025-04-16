FROM ghcr.io/astral-sh/uv:debian-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    git 

COPY . /app

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/opt/venv

RUN uv sync --frozen

CMD ["uv", "run", "copick_server/server.py", "example_copick.json", "--host", "0.0.0.0", "--port", "8012"]