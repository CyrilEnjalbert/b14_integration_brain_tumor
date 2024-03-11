# Use an existing base image with Python 3.11.5
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Copy the current directory contents into the container at /app
COPY . .

# Install dependencies
RUN pip3 install --upgrade pip 


# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install MLflow
RUN pip3 install mlflow

RUN sudo apt-get install libgl1-mesa-glx

# Install FastAPI and pymongo for MongoDB
RUN pip3 install fastapi uvicorn pymongo

# Expose MLflow, FastAPI, and MongoDB ports
EXPOSE 5000 8000 27017

# Define environment variable
ENV MLFLOW_SERVER_HOST 0.0.0.0
ENV MLFLOW_SERVER_PORT 5000
ENV MONGO_HOST mongodb://mongo:27017

# Command to run the application when the container starts
CMD ["sh", "-c", "mlflow ui --port $MLFLOW_SERVER_PORT & python3 train_and_log_model.py & uvicorn model_api:app --host 0.0.0.0 --port 8000"]