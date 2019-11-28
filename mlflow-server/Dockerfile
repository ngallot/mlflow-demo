FROM python:3.7-slim-stretch

# Install mlflow
RUN python -m pip install --upgrade pip mlflow==1.4.0

# Create default stores
RUN mkdir /backend_store
RUN mkdir /artifact_root


# Expose mlflow port
EXPOSE 1234

# Define entry point
ENTRYPOINT mlflow server --host 0.0.0.0 --port 1234 --backend-store-uri /backend_store --default-artifact-root /artifact_root