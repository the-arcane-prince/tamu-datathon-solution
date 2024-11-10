FROM python:3.10.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v3.27.4/cmake-3.27.4-linux-x86_64.tar.gz && \
    tar -xzvf cmake.tar.gz --strip-components=1 -C /usr/local && \
    rm cmake.tar.gz

RUN cmake --version

# Install dependencies first to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from current directory
COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5000"]
