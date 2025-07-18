FROM python:3.10-slim

WORKDIR /app

COPY ./app /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
