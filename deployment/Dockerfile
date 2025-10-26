FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (can be overridden by cloud platform)
EXPOSE 8080

# Set default environment variables
ENV PORT=8080
ENV FLASK_ENV=production

# Run the application
CMD exec gunicorn --bind :$PORT --workers 4 --threads 8 --timeout 120 app.main:app
