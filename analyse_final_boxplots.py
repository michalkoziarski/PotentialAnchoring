import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from extract_dataset_info import extract
from pathlib import Path


RESULTS_PATH = Path(__file__).parent / 'results'
VISUALIZATIONS_PATH = Path(__file__).parent / 'visualizations'


if __name__ == '__main__':
    for di in [None, 0.6]:
        for classifier in ['CART', 'KNN', 'SVM', 'MLP']:
            csv_path = RESULTS_PATH / 'results_final.csv'

            try:
                info = pd.read_csv(RESULTS_PATH / 'dataset_info.csv')
            except FileNotFoundError:
                print('Extracting dataset info...')

                info = extract(verbose=False)

            if di is not None:
                info = info[info['DI'] >= di]

            datasets = info['Name']

            df = pd.read_csv(csv_path)
            df = df.groupby(['Dataset', 'Classifier', 'Resampler', 'Metric'])['Score'].agg('mean').reset_index()
            df = df[df['Metric'] == 'G-mean']
            df = df[df['Classifier'] == classifier]
            df = df[df['Dataset'].isin(datasets)]
            df['Resampler'] = df['Resampler'].replace({
                'polynom-fit-SMOTE': 'pf-SMOTE',
                'Assembled-SMOTE': 'A-SMOTE',
                'SMOTE-TomekLinks': 'SMOTE-TL'
            })

            order = [
                'SMOTE', 'pf-SMOTE', 'Lee', 'SMOBD',
                'G-SMOTE', 'LVQ-SMOTE', 'A-SMOTE',
                'SMOTE-TL', 'RBO', 'PA'
            ]

            g = sns.boxplot(y='Resampler', x='Score', data=df, palette='muted', order=order, orient='h')
            g.set(xlabel=None, ylabel=None)

            if di is None:
                file_name = f'boxplot_{classifier}_all.pdf'
            else:
                file_name = f'boxplot_{classifier}_{di}.pdf'

            plt.savefig(VISUALIZATIONS_PATH / file_name, bbox_inches='tight')
            plt.close()
