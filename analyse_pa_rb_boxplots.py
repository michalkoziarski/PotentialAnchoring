import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


RESULTS_PATH = Path(__file__).parent / 'results'
VISUALIZATIONS_PATH = Path(__file__).parent / 'visualizations'


if __name__ == '__main__':
    for classifier in ['CART', 'KNN', 'SVM', 'MLP']:
        csv_path = RESULTS_PATH / 'results_pa_rb_comparison.csv'

        df = pd.read_csv(csv_path)
        df = df.groupby(['Dataset', 'Classifier', 'Resampler', 'Metric'])['Score'].agg('mean').reset_index()
        df = df[df['Metric'] == 'G-mean']
        df = df[df['Classifier'] == classifier]
        df['Resampler'] = df['Resampler'].replace({
            'polynom-fit-SMOTE': 'pf-SMOTE',
            'Assembled-SMOTE': 'A-SMOTE',
            'SMOTE-TomekLinks': 'SMOTE-TL'
        })

        order = [
            'RBO', 'PAO', 'RBU', 'PAU'
        ]

        g = sns.boxplot(x='Resampler', y='Score', data=df, palette='muted', order=order)
        g.set(xlabel=None, ylabel=None)

        VISUALIZATIONS_PATH.mkdir(exist_ok=True, parents=True)

        plt.savefig(VISUALIZATIONS_PATH / f'pa_rb_boxplot_{classifier}.pdf', bbox_inches='tight')
        plt.close()
