FROM huggingface/text-generation-inference:1.3.0

# Install additional requirements
COPY app/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Copy the handler code
COPY app /app

# Set environment variables
ENV HUGGINGFACE_HUB_CACHE=/data
ENV TRANSFORMERS_CACHE=/data

# Use custom handler
ENV HANDLER_PATH=/app/handler.py