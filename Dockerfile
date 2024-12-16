FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim 
# Set environment variables
ENV PORT=7860

WORKDIR /var/task

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir  -r requirements.txt

# Copy application code and model
COPY app.py ./
COPY icons.py ./
#COPY traced_models/ ./traced_models/
COPY onxx_models/ ./onxx_models/
COPY sample_data/ ./sample_data/
EXPOSE 7860
# Set command
CMD exec uvicorn --host 0.0.0.0 --port $PORT app:app 

