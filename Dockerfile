# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /code

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application - this will be overridden by docker-compose for the 'app' service
# But it's good practice to have a default for direct `docker run`
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
