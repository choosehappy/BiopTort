FROM python:3.9

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" --no-install-recommends \
    build-essential \
    cmake \
    ca-certificates \
    python3-venv \
    python3-wheel \
    python3-dev \
    python3-setuptools \
    libglib2.0-0 \
    libopenslide0 \
    procps \
    # Requirement for opencv
    libgl1 

RUN mkdir -p /opt/BiopTort
WORKDIR /opt/BiopTort
COPY . /opt/BiopTort/

ENV PATH="/opt/BiopTort/venv/bin:$PATH"

RUN python3.9 -m venv venv \
    && python3.9 -m pip install --upgrade pip \
    && pip install -e .
