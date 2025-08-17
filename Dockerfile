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

# Copy all application files
COPY app.py .
COPY train_model.py .
COPY Crop_recommendation.csv .
COPY templates/ ./templates/
COPY static/ ./static/

# NEW: Run the training script to generate model files before starting
RUN python train_model.py

EXPOSE 8080

# The Start Command (or Docker Command on Render)
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
