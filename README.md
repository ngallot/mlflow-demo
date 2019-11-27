# mlflow-demo
A demo project using MLflow machine learning lifecycle management platform.
We use here a public Kaggle dataset, and we're building an ML model for predicting a pulsar star.


### Data
You can find the data and its description [here](https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star/version/1#).
You can create a directory named data at the root of your project and download the contents of the CSV in it.
```bash
mkdir ./data
wget https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star/download/bx3a6Cxv6tjlN5yLnEdB%2Fversions%2FR1INhXqu2MVIqPl6xOVA%2Ffiles%2Fpulsar_stars.csv?datasetVersionNumber=1 -o ./data/pulsar_stars.csv


```

### Project setup
To run this project, it is recommended to first setup a virtual environment and install the requirements.txt in it.
```bash
virtual env -p path/to/your/python venv
source venv/bin/activate
pip install -r requirements.txt
```

