FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

ADD scripts/install_requirements.sh scripts/requirements.txt /setup/

RUN apt-get update -y \
    && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 python3-pip \
    && /bin/bash /setup/install_requirements.sh \
    && mkdir -p /app/data

COPY ./ /app/

WORKDIR /app

ENTRYPOINT ["python3", "main.py"]

