# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

ARG PYTHON_VERSION=3.9.18
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN apt-get -y update
# for dlib
RUN apt-get install -y build-essential cmake

#make sure opencv is installed 
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8501

# Run the application.
CMD python -m streamlit run ./run/Run_streamlit.py

# Tells Docker how to test a container to check that it is still working. Your container needs to listen to Streamlit’s (default) port 8501
#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Configure a container that will run as an executable
#ENTRYPOINT ["streamlit", "run", "./run/Run_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
