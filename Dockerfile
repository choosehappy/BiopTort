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

RUN python3.9 -m pip install uv

RUN mkdir -p /opt/BiopTort
WORKDIR /opt/BiopTort

RUN uv venv /opt/BiopTort/.venv
ENV VIRTUAL_ENV=/opt/BiopTort/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ../pyproject.toml /opt/BiopTort/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r pyproject.toml

COPY ./ /opt/BiopTort/
# Install BiopTort to the uv-managed environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install .
