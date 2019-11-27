import sys
import os
import argparse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
import logging


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('psp.mltraining')


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
    pred = model.predict(X_test)
    return auc_score(pred, y_test)


# @ignore_warnings(category=ConvergenceWarning)
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

    # Get best model from hyper parameters search and log model params in mlflow
    model = pipeline.steps[1][1]
    best_params = model.best_params_

    logger.info('Training done ')

    # Logging results to mlflow tracking
    # mlflow_tracking_url = 'http://0.0.0.0:5555'
    # print(f'Setting mlflow tracking url {mlflow_tracking_url}')
    # mlflow.set_tracking_uri(mlflow_tracking_url)
    # experiment_name = 'pulsar_stars_training'
    # experiment_id = create_experiment(experiment_name)
    # print(f'Setting experiment {experiment_name} (id={experiment_id}) for mlflow')
    # with mlflow.start_run(experiment_id=experiment_id, run_name='training'):
    #     mlflow.log_params(model.best_params_)
    #     mlflow.log_metric('auc_test', auc_test)
    #     mlflow.sklearn.log_model(pipeline, 'model')


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

    train(args.data_path, float(args.test_size))
