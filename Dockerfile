# Use an official PyTorch Lightning runtime as a parent image
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.11-torch2.3-cuda12.1.0

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --ignore-installed --upgrade -r /code/requirements.txt
