FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir --no-deps xgboost==3.3.0
RUN pip install --no-cache-dir gunicorn==23.0.0

COPY app.py /app/
COPY scripts/ /app/scripts/
COPY templates/ /app/templates/
COPY docs/ /app/docs/
COPY data/ /app/data/
COPY output/ /app/output/
COPY mobile_app/build/web /app/mobile_app/build/web

ENV PORT=5000
EXPOSE 5000

CMD gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} app:app
