FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir mlflow tensorflow scikit-learn

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
