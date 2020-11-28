import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from extract_dataset_info import extract
from pathlib import Path
from schedule_final import RESAMPLERS
from scipy.stats import pearsonr


CLASSIFIERS = ['CART', 'KNN', 'SVM', 'MLP']
METRICS = ['Precision', 'Recall', 'AUC', 'G-mean']
P_VALUE = 0.10
RESULTS_PATH = Path(__file__).parent / 'results'
VISUALIZATIONS_PATH = Path(__file__).parent / 'visualizations'


def load_final_dict(classifier, metric):
    csv_path = RESULTS_PATH / 'results_final.csv'

    df = pd.read_csv(csv_path)
    df = df[(df['Classifier'] == classifier) & (df['Metric'] == metric)]

    df = df.groupby(
        ['Dataset', 'Classifier', 'Resampler', 'Metric']
    )['Score'].agg('mean').reset_index()

    rows = []

    for dataset in datasets.names():
        row = [dataset]

        for resampler in RESAMPLERS:
            ds = df[(df['Resampler'] == resampler) & (df['Dataset'] == dataset)]

            assert len(ds) == 1

            row.append(np.round(list(ds['Score'])[0], 4))

        rows.append(row)

    return pd.DataFrame(rows, columns=['Dataset'] + RESAMPLERS)


def prepare_df():
    try:
        info = pd.read_csv(RESULTS_PATH / 'dataset_info.csv')
    except FileNotFoundError:
        print('Extracting dataset info...')

        info = extract(verbose=False)

    info = info[info['Name'].isin(datasets.names())].reset_index(drop=True)
    info['Dataset'] = info['Name']
    info = info.drop('Name', axis=1)

    rows = []

    for clf in CLASSIFIERS:
        for metric in METRICS:
            res = load_final_dict(clf, metric)
            res['Rank'] = list(res.rank(axis=1, ascending=False)['PA'])

            res = pd.merge(res, info, on='Dataset')

            assert len(res) == 60

            for _, row in res.iterrows():
                rows.append([row['Dataset'], clf, metric, row['Rank'], row['DI']])

    return pd.DataFrame(rows, columns=['Dataset', 'Classifier', 'Metric', 'Rank', 'DI'])


def export_correlation(df):
    print(' & '.join([''] + CLASSIFIERS) + ' \\\\')
    print('\\midrule')

    for metric in METRICS:
        row = [metric]

        for clf in CLASSIFIERS:
            ds = df[(df['Classifier'] == clf) & (df['Metric'] == metric)]

            x = list(ds['DI'])
            y = list(ds['Rank'])

            rho, pval = pearsonr(x, y)

            col = '%+.4f' % np.round(rho, 4)

            if pval <= P_VALUE:
                col = '\\textbf{%s}' % col

            row.append(col)

        print(' & '.join(row) + ' \\\\')


def visualize(df):
    g = sns.lmplot(
        x='DI', y='Rank', data=df,
        col='Metric', row='Classifier', hue='Metric', truncate=True,
        sharex=False, height=3.0, scatter_kws={'alpha': 0.3, 's': 100}
    )
    g.set(ylim=(0.5, 10.5))
    g.set(xlim=(0.05, 1.05))
    g.set_titles('{row_name}, {col_name}')

    VISUALIZATIONS_PATH.mkdir(exist_ok=True, parents=True)

    plt.savefig(VISUALIZATIONS_PATH / 'di_impact.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    df = prepare_df()
    export_correlation(df)
    visualize(df)
