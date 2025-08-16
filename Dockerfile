# Stage 1: The "builder" stage with all the tools
FROM python:3.13 as builder

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---
# Stage 2: The "final" stage for a slim, efficient container
FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

COPY app.py .
COPY train_model.py .
COPY crop_model.joblib .
COPY crop_conditions.csv .
COPY Crop_recommendation.csv .
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 8080

# UPDATED: We now run gunicorn as a Python module, which is more reliable.
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:8080", "app:app"]