# Hugging Face Spaces deployment — CPU only
FROM python:3.11-slim

WORKDIR /code

# CPU-only PyTorch (much smaller than the default CUDA build)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    fastapi \
    "uvicorn[standard]"

# Application code
COPY config.py data.py model.py server.py /code/
COPY web/         /code/web/
COPY checkpoints/ /code/checkpoints/

# HF Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
