FROM python:3.11-slim

COPY model.cbm /model.cbm
COPY app.py /app.py

RUN pip install --no-cache-dir numpy catboost

CMD ["python", "app.py"]