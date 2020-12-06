import argparse
import datasets
import logging
import metrics
import numpy as np
import pandas as pd

from algorithms import PAO, PAU
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def evaluate_trial(n_anchors, fold):
    RESULTS_PATH = Path(__file__).parents[0] / 'results_n_anchors'
    RANDOM_STATE = 42

    for resampler_name in ['PAO', 'PAU']:
        for dataset_name in datasets.names():
            classifiers = {
                'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),
                'KNN': KNeighborsClassifier(n_neighbors=3),
                'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE),
                'MLP': MLPClassifier(random_state=RANDOM_STATE)
            }

            trial_name = f'{dataset_name}_{fold}_{resampler_name}_{n_anchors}'
            trial_path = RESULTS_PATH / f'{trial_name}.csv'

            if trial_path.exists():
                continue

            logging.info(f'Evaluating {trial_name}...')

            dataset = datasets.load(dataset_name)

            (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

            resamplers = {
                'PAO': PAO(n_anchors=n_anchors, random_state=RANDOM_STATE),
                'PAU': PAU(n_anchors=n_anchors, random_state=RANDOM_STATE)
            }

            resampler = resamplers[resampler_name]

            assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

            try:
                X_train, y_train = resampler.sample(X_train, y_train)
            except RuntimeError:
                continue

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
                    row = [dataset_name, fold, classifier_name, resampler_name, n_anchors, scoring_function_name, score]
                    rows.append(row)

            columns = ['Dataset', 'Fold', 'Classifier', 'Resampler', 'Anchors', 'Metric', 'Score']

            RESULTS_PATH.mkdir(exist_ok=True, parents=True)

            pd.DataFrame(rows, columns=columns).to_csv(trial_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-fold', type=int)
    parser.add_argument('-n_anchors', type=int)

    args = parser.parse_args()

    evaluate_trial(args.n_anchors, args.fold)
