FROM jayaneetha/images:tf2.1.0-gpu-py3.6.8-base

WORKDIR /app

COPY rl .
COPY *.py .
COPY tf210.txt .

RUN mkdir -p rl-files/logs && sudo chown user:user rl-files/logs
RUN mkdir -p rl-files/models && sudo chown user:user rl-files/models

RUN pip install -r tf210.txt
