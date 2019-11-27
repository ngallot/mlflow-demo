# mlflow-demo
A demo project using MLflow machine learning lifecycle management platform.
We use here a public Kaggle dataset, and we're building an ML model for predicting a pulsar star.


### Data
You can find the data and its description [here](https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star/version/1#).
You can create a directory named data at the root of your project and download the contents of the CSV in it, the unzip it.

### Project setup
To run this project, it is recommended to first setup a virtual environment and install the requirements.txt in it.
```bash
virtual env -p path/to/your/python venv
source venv/bin/activate
pip install -r requirements.txt
```

### Setup local mlflow tracking server
In this section, we will build a Docker container exposing the mlflow-tracking server api. It will allow us to update our training script in order to log metrics and
models in mlflow.

The contents of this server is very simple: just a Dockerfile with very few instructions.

```dockerfile
FROM python:3.7-slim-stretch

# Install mlflow
RUN python -m pip install --upgrade pip mlflow==1.4.0

# Expose mlflow port
EXPOSE 1234

# Define entry point
ENTRYPOINT mlflow server --host 0.0.0.0 --port 1234
```
Then we define a docker-compose file at the project root to build this service:

```yaml
version: '3.1'

services:
  mlflow-server:
    build:
      context: mlflow-server
      dockerfile: Dockerfile
    image:
      mlflow:1.4.0
    ports:
      - "1234:1234"
```

To start the mlflow server, we just have to run the command: 
```bash
docker-compose up --build mlflow-server
```
It will expose the mlflow api at http://localhost:1234

You can the easily test the server using mlflow command line (from your venv):
```bash
export MLFLOW_TRACKING_URI=http://localhost:1234
mlflow experiments create -n test-server
```

Then you can see your created experiment:
```bash
mlflow experiments list

# Experiment Id  Name         Artifact Location
#---------------  -----------  -----------------------------------------------
#              0  Default      /home/**/mlflow-demo/mlruns/0
#              1  test-server  /home/**/mlflow-demo/mlruns/1
```

You can also visualize it in the mlflow UI by opening a browser to http://localhost:1234:

<p align="center">
  <img src="img/mlflow-ui.png" alt="MLflow UI" />
</p>

This experiment was just for testing so we can now delete it: 
```bash
mlflow experiments delete -x 1
```