import argparse
import datasets
import logging
import metrics
import numpy as np
import pandas as pd

from algorithms import CPA
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def evaluate_trial(ratio, fold):
    RESULTS_PATH = Path(__file__).parents[0] / 'results_preliminary_cpa'
    RANDOM_STATE = 42

    for dataset_name in datasets.names():
        classifiers = {
            'CART': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'L-SVM': LinearSVC(random_state=RANDOM_STATE),
            'R-SVM': SVC(random_state=RANDOM_STATE, kernel='rbf'),
            'P-SVM': SVC(random_state=RANDOM_STATE, kernel='poly'),
            'LR': LogisticRegression(random_state=RANDOM_STATE),
            'NB': GaussianNB(),
            'R-MLP': MLPClassifier(random_state=RANDOM_STATE),
            'L-MLP': MLPClassifier(random_state=RANDOM_STATE, activation='identity')
        }

        trial_name = f'{dataset_name}_{fold}_{ratio}'
        trial_path = RESULTS_PATH / f'{trial_name}.csv'

        if trial_path.exists():
            continue

        logging.info(f'Evaluating {trial_name}...')

        dataset = datasets.load(dataset_name)

        (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

        resampler = CPA(ratio=ratio, random_state=RANDOM_STATE)

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
                'Specificity': metrics.specificity,
                'AUC': metrics.auc,
                'G-mean': metrics.g_mean,
                'F-measure': metrics.f_measure
            }

            for scoring_function_name in scoring_functions.keys():
                score = scoring_functions[scoring_function_name](y_test, predictions)
                row = [dataset_name, fold, classifier_name, ratio, scoring_function_name, score]
                rows.append(row)

        columns = ['Dataset', 'Fold', 'Classifier', 'Ratio', 'Metric', 'Score']

        RESULTS_PATH.mkdir(exist_ok=True, parents=True)

        pd.DataFrame(rows, columns=columns).to_csv(trial_path, index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-fold', type=int)
    parser.add_argument('-ratio', type=float)

    args = parser.parse_args()

    evaluate_trial(args.ratio, args.fold)
