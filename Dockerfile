FROM jayaneetha/images:tf2.1.0-gpu-py3.6.8-base

RUN sudo apt update && sudo apt install -y nano

WORKDIR /app

COPY tf210.txt .
RUN pip install -r tf210.txt

COPY rl /app/rl
COPY *.py ./

RUN sudo chown -R user rl
RUN sudo chown user *.py


#CMD ["python rl_run.py --data-version=esd --policy=ZetaPolicy --pre-train-dataset=esd --pre-train=true --env-name=Zeta2.0 --disable-wandb=True"]