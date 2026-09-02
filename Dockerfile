# Repuragent Docker Container
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    JAVA_HOME=/usr/lib/jvm/default-java

ENV USER=repuragent
ENV HOME=/home/$USER
ENV PERSIST_ROOT=/home/$USER/app/persistence


RUN useradd -m -u 1000 $USER

# Install system dependencies including Java 11
RUN apt-get update && apt-get install -y \
    # Java 11 for CPSign
    default-jdk \
    # Build tools and system utilities
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    wget \
    unzip \
    # For some Python packages that might need compilation
    python3-dev \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify Java installation
RUN java -version

# Set working directory
WORKDIR $HOME/app

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-fetch Chroma default embedding model to avoid runtime download delays
RUN python - <<'PY'
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

embedder = DefaultEmbeddingFunction()
embedder(["warmup"])  # triggers model download & caching
print("Chroma default embedding model cached during build.")
PY

# Copy the entire application
COPY . .

# Set proper permissions
RUN chmod +x models/CPSign/cpsign-2.0.0-fatjar.jar 2>/dev/null || true
RUN chmod -R 755 $HOME/app
RUN chown -R $USER:$USER $HOME

# Create volumes for persistent data and memory stores
VOLUME ["/home/repuragent/app/persistence"]

# Expose Gradio port
EXPOSE 7860

# Default command
USER $USER

CMD ["python", "main.py"]
