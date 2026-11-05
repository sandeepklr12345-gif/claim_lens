# CPU image for the API and the UI. Ollama stays on the host (SPEC.md 13.5).
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface
WORKDIR /app

COPY requirements.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY claimlens/ claimlens/
COPY api/ api/
COPY app/ app/
COPY configs/ configs/
COPY data/ data/
COPY .streamlit/ .streamlit/

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
