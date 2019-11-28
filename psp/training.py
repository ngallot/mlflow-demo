import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment, RunInfo
import mlflow.sklearn

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('psp.mltraining')


def get_or_create_experiment(experiment_name) -> Experiment:
    """
    Creates an mlflow experiment
    :param experiment_name: str. The name of the experiment to be set in MLFlow
    :return: the experiment created if it doesn't exist, experiment if it is already created.
    """
    try:
        client = MlflowClient()
        experiment: Experiment = client.get_experiment_by_name(name=experiment_name)
        if experiment and experiment.lifecycle_stage != 'deleted':
            return experiment
        else:
            experiment_id = client.create_experiment(name=experiment_name)
            return client.get_experiment(experiment_id=experiment_id)
    except Exception as e:
        logger.error(f'Unable to get or create experiment {experiment_name}: {e}')


def auc_score(y_pred, y_true):
    """
    Computes the AUC metric between model predictions and knows labels
    :param y_pred: the predictions output by the model
    :param y_true: the ground truth (aka known labels)
    :return: the Area Under the Roc Curve
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)


def auc_score_model(model, X_test, y_test):
    """
    Computes the AUC score of a model on a test dataset
    :param model: a ML model with a 'predict' method
    :param X_test: the test features
    :param y_test: the test labels
    :return: the AUC between model predictions and ground truth
    """
    return auc_score(model.predict(X_test), y_test)


def train(data_path: str, test_size: float):

    np.random.seed(40)

    # Load data and define features and target
    ps = pd.read_csv(data_path)
    features = [c for c in ps.columns if c != 'target_class']
    X = ps[features].values
    y = ps['target_class'].values
    logger.info(f'Data shape: {ps.shape}')

    # Split data in train / test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    logger.info(f"""
X_train size: {X_train.shape}
X_test size: {X_test.shape}
    """)

    # Define pipeline and run cross validated hyper parameters search
    scaler = StandardScaler()
    gs_params = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': np.linspace(start=1, stop=20, num=2),
        'class_weight': [None, 'balanced'],
        'l1_ratio': np.linspace(start=0.1, stop=0.9, num=3)
    }

    lr = LogisticRegression(solver='saga')
    lr_cv = GridSearchCV(cv=5, estimator=lr, param_grid=gs_params, n_jobs=-1, scoring='roc_auc')

    pipeline = Pipeline(steps=[('scaler', scaler), ('logistic_regression_cv', lr_cv)])

    logger.info('Starting model training...')
    pipeline.fit(X_train, y_train)

    logger.info('Training done, starting model evaluation...')
    auc_test = auc_score_model(pipeline, X_test, y_test)
    logger.info(f'AUC score with hyper params search and cross validated logistic regression: {auc_test}')

    return pipeline, 'auc_test', auc_test


def log_metrics_and_model(pipeline, test_metric_name: str, test_metric_value: float):

    experiment_name = 'pulsar_stars_training'
    experiment: Experiment = get_or_create_experiment(experiment_name)

    model = pipeline.steps[1][1]
    best_params = model.best_params_

    def build_local_model_path(relative_artifacts_uri: str, model_name: str):
        import os
        absolute_artifacts_uri = relative_artifacts_uri.replace('./', f'{os.getcwd()}/')
        return os.path.join(absolute_artifacts_uri, model_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='training') as run:
        model_name = 'sklearn_logistic_regression'
        run_info: RunInfo = run.info
        mlflow.log_params(best_params)
        mlflow.log_metric(test_metric_name, test_metric_value)
        mlflow.sklearn.log_model(pipeline, model_name)
        model_path = build_local_model_path(run_info.artifact_uri, model_name)

        logger.info(f'Model for run_id {run_info.run_id} exported at {model_path}')


if __name__ == '__main__':
    """
    Lauches the training script
    :param data_path: the path to the full dataset
    :param test_size: the ratio of test data vs training data
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path')
    parser.add_argument('--test-size')
    args = parser.parse_args()

    if not args.data_path:
        raise Exception('argument --data-path should be specified')

    if not args.test_size:
        raise Exception('argument --test-size should be specified')

    pipeline, test_metric_name, test_metric_value = train(args.data_path, float(args.test_size))
    if os.getenv('MLFLOW_TRACKING_URI', None):
        log_metrics_and_model(pipeline=pipeline, test_metric_name=test_metric_name, test_metric_value=test_metric_value)
    else:
        logger.info(f'Env var MLFLOW_TRACKING_URI not set, skipping mlflow logging.')
