FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-train the Random Forest model during image build for instant startup
RUN python cli.py train

EXPOSE 7860
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --workers 2 --timeout 120"]
