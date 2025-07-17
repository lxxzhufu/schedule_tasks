FROM docker.1ms.run/python:3.12-slim-bookworm 
ENV PATH="/opt/ffmpeg-master-latest-linux64-gpl/bin:${PATH}"
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY ffmpeg-master-latest-linux64-gpl /opt/ffmpeg-master-latest-linux64-gpl 
# Copy the project into the image 
ADD . /app 
# Sync the project into a new environment, using the frozen lockfile 
WORKDIR /app 
RUN uv sync --frozen # 最基本的启动 
# Presuming there is a `my_app` command provided by the project 
CMD ["uv", "run", "main.py"]