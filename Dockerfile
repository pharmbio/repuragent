# Repuragent Docker Container
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    JAVA_HOME=/usr/lib/jvm/default-java

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
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/results /app/models

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Set proper permissions
RUN chmod +x models/CPSign/cpsign-2.0.0-fatjar.jar 2>/dev/null || true
RUN chmod -R 755 /app

# Create a volume for persistent data
VOLUME ["/app/data", "/app/results"]

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Create entrypoint script
COPY <<EOF /app/entrypoint.sh
#!/bin/bash
set -e

# Check if .env file exists, if not create a template
if [ ! -f /app/.env ]; then
    echo "Creating .env template..."
    cat > /app/.env << 'ENVEOF'
# OpenAI API key (mandatory)
OPENAI_API_KEY=your-api-key

# LANGSMITH TRACING API KEY (Optional)
#LANGSMITH_TRACING=true
#LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
#LANGSMITH_API_KEY=tracking-api-key
#LANGSMITH_PROJECT=project-name
ENVEOF
    echo "Please edit /app/.env and set your OPENAI_API_KEY"
fi

# Check Java installation
echo "Java version:"
java -version

# Check CPSign jar file
if [ -f "/app/models/CPSign/cpsign-2.0.0-fatjar.jar" ]; then
    echo "CPSign jar file found"
else
    echo "Warning: CPSign jar file not found at /app/models/CPSign/cpsign-2.0.0-fatjar.jar"
fi

# Check if models directory has the required model files
if [ -d "/app/models" ] && [ "$(ls -A /app/models)" ]; then
    echo "Model files found: \$(ls /app/models | wc -l) files"
else
    echo "Warning: No model files found in /app/models"
fi

# Start the application
echo "Starting Repuragent application..."
cd /app
streamlit run main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
EOF

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]