FROM python:3.11-slim

WORKDIR /app

# build-essential is needed for some sentence-transformers/torch deps to build wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Graded baseline runs fully offline -- MOCK_LLM defaults to "1" here.
# Override with -e MOCK_LLM=0 (and -e GROQ_API_KEY=...) only for the
# optional, ungraded real-LLM extension.
ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
