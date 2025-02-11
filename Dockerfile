# Use the official Hugging Face TGI container
FROM ghcr.io/huggingface/text-generation-inference:latest

# Let Hugging Face Inference Endpoints know where your custom handler is:
ENV HANDLER_PATH=/app/handler.py

# If you want your code to see GPU in Docker, you need an nvidia/cuda-based environment,
# but TGI’s official image already has that. We do *not* override the entrypoint.

# Create any folders your scripts expect to exist at runtime:
RUN mkdir -p /embeddings/subchunked /tmp/text-generation-server

# Copy in your Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy in your code
COPY app /app
COPY chapter2 /chapter2
COPY ems /ems

# We do *not* override the entrypoint or re-install text-generation.
# TGI’s built-in entrypoint will launch the server & import `handler.py`.

# For Hugging Face Endpoints: set CMD to empty, so TGI gets all the default args it needs.
CMD []

#FNORD
