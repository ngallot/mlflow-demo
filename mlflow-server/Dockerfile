FROM python:3.7-slim-stretch

# Install mlflow
RUN python -m pip install --upgrade pip mlflow==1.4.0

# Expose mlflow port
EXPOSE 1234

# Define entry point
ENTRYPOINT mlflow server --host 0.0.0.0 --port 1234