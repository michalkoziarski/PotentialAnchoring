import datasets
import numpy as np
import pandas as pd

from collections import OrderedDict
from pathlib import Path
from scipy.stats import wilcoxon


ALGORITHMS = ['RBO', 'PAO', 'RBU', 'PAU']
CLASSIFIERS = ['CART', 'KNN', 'SVM', 'MLP']
METRICS = ['AUC', 'G-mean']
P_VALUE = 0.10
RESULTS_PATH = Path(__file__).parent / 'results'


def load_final_dict(classifier, metric):
    csv_path = RESULTS_PATH / 'results_pa_rb_comparison.csv'

    df = pd.read_csv(csv_path)
    df = df[(df['Classifier'] == classifier) & (df['Metric'] == metric)]

    measurements = OrderedDict()

    for algorithm in ALGORITHMS:
        measurements[algorithm] = []

        for dataset in datasets.names():
            scores = df[(df['Resampler'] == algorithm) & (df['Dataset'] == dataset)]['Score']

            assert len(scores) == 10

            measurements[algorithm].append(np.mean(scores))

    return measurements


if __name__ == '__main__':
    for rb_name, pa_name in zip(['RBO', 'RBU'], ['PAO', 'PAU']):
        print('& \\multicolumn{3}{l}{' + METRICS[0] + '} & \\multicolumn{3}{l}{' + METRICS[1] + '} \\\\')
        print('\\cmidrule(l){2-4} \\cmidrule(l){5-7}')
        print(f'Clf. & {rb_name} & {pa_name} & $p$-value & {rb_name} & {pa_name} & $p$-value \\\\')

        for classifier in CLASSIFIERS:
            row = [classifier]

            for metric in METRICS:
                d = load_final_dict(classifier, metric)

                x = d[rb_name]
                y = d[pa_name]

                p = np.round(wilcoxon(x, y)[1], 4)

                rb_wins = 0
                pa_wins = 0
                ties = 0

                for rb_i, pa_i in zip(x, y):
                    if rb_i > pa_i:
                        rb_wins += 1
                    elif pa_i > rb_i:
                        pa_wins += 1
                    else:
                        ties += 1

                if p <= P_VALUE:
                    p = '\\textbf{' + f'{p:.4f}' + '}'
                else:
                    p = f'{p:.4f}'

                row += [rb_wins, pa_wins, p]

            print(' & '.join([str(r) for r in row]) + ' \\\\')

        print('')
