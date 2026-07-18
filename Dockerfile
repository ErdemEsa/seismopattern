FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir --no-deps xgboost==3.3.0

COPY app.py /app/
COPY scripts/ /app/scripts/
COPY templates/ /app/templates/
COPY docs/ /app/docs/

EXPOSE 5000

CMD ["python", "app.py"]