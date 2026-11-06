FROM ultralytics/ultralytics:latest-jetson-jetpack4

WORKDIR /workspace

COPY requirements.txt .

# System packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    cmake \
    pkg-config \
    curl \
    git

# Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# maturin
RUN pip3 install --upgrade pip
RUN pip3 install maturin

# mọi package TRỪ DeepFilterNet
RUN grep -v deepfilternet requirements.txt > req_no_df.txt && \
    pip3 install --no-cache-dir -r req_no_df.txt

# cuối cùng mới DeepFilterNet
RUN pip3 install --no-build-isolation deepfilternet==0.5.6

COPY . .

CMD ["bash"]