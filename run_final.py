import argparse
import datasets
import logging
import metrics
import numpy as np
import pandas as pd
import smote_variants as sv

from algorithms import PA
from pathlib import Path
from rbo import RBO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def evaluate_trial(resampler_name, fold):
    RESULTS_PATH = Path(__file__).parents[0] / 'results_final'
    RANDOM_STATE = 42

    resamplers = {
        'SMOTE': sv.distance_SMOTE(random_state=RANDOM_STATE),
        'polynom-fit-SMOTE': sv.polynom_fit_SMOTE(random_state=RANDOM_STATE),
        'Lee': sv.Lee(random_state=RANDOM_STATE),
        'SMOBD': sv.SMOBD(random_state=RANDOM_STATE),
        'G-SMOTE': sv.G_SMOTE(random_state=RANDOM_STATE),
        'LVQ-SMOTE': sv.LVQ_SMOTE(random_state=RANDOM_STATE),
        'Assembled-SMOTE': sv.Assembled_SMOTE(random_state=RANDOM_STATE),
        'SMOTE-TomekLinks': sv.SMOTE_TomekLinks(random_state=RANDOM_STATE),
        'RBO': RBO(random_state=RANDOM_STATE),
        'PA': PA(random_state=RANDOM_STATE)
    }

    for dataset_name in datasets.names():
        classifiers = {
            'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'SVM': SVC( kernel='rbf', random_state=RANDOM_STATE),
            'MLP': MLPClassifier(random_state=RANDOM_STATE)
        }

        trial_name = f'{dataset_name}_{fold}_{resampler_name}'
        trial_path = RESULTS_PATH / f'{trial_name}.csv'

        if trial_path.exists():
            continue

        logging.info(f'Evaluating {trial_name}...')

        dataset = datasets.load(dataset_name)

        (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

        resampler = resamplers[resampler_name]

        assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

        X_train, y_train = resampler.sample(X_train, y_train)

        rows = []

        for classifier_name in classifiers.keys():
            classifier = classifiers[classifier_name]

            clf = classifier.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            scoring_functions = {
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'AUC': metrics.auc,
                'G-mean': metrics.g_mean
            }

            for scoring_function_name in scoring_functions.keys():
                score = scoring_functions[scoring_function_name](y_test, predictions)
                row = [dataset_name, fold, classifier_name, resampler_name, scoring_function_name, score]
                rows.append(row)

        columns = ['Dataset', 'Fold', 'Classifier', 'Resampler', 'Metric', 'Score']

        RESULTS_PATH.mkdir(exist_ok=True, parents=True)

        pd.DataFrame(rows, columns=columns).to_csv(trial_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-fold', type=int)
    parser.add_argument('-resampler_name', type=str)

    args = parser.parse_args()

    evaluate_trial(args.resampler_name, args.fold)
