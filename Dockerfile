FROM python:3.9-bookworm

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" --no-install-recommends \
    build-essential \
    cmake \
    ca-certificates \
    python3-venv \
    python3-wheel \
    python3-dev \
    python3-setuptools \
    libopenslide0


RUN mkdir -p /opt/BiopTort
WORKDIR /opt/BiopTort
COPY . /opt/BiopTort/

ENV PATH="/opt/BiopTort/venv/bin:$PATH"

RUN python3 -m venv venv \
    && python3 -m pip install --upgrade pip \
    && pip install -e .

