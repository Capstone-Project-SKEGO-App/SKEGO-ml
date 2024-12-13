# Gunakan image Python yang kompatibel
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy semua file ke container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn
RUN pip install gunicorn

# Expose port Flask
EXPOSE 8080

# Jalankan aplikasi Flask dengan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "main:app"]
