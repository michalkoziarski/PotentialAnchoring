import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


RESULTS_PATH = Path(__file__).parent / 'results'
VISUALIZATIONS_PATH = Path(__file__).parent / 'visualizations'


for classifier in ['CART', 'KNN', 'SVM', 'MLP']:
    csv_path = RESULTS_PATH / 'results_final.csv'

    df = pd.read_csv(csv_path)
    df = df.groupby(['Dataset', 'Classifier', 'Resampler', 'Metric'])['Score'].agg('mean').reset_index()
    df = df[df['Metric'] == 'G-mean']
    df = df[df['Classifier'] == classifier]
    df['G-mean'] = df['Score']
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

    sns.boxplot(y='Resampler', x='G-mean', data=df, palette='muted', order=order, orient='h')

    plt.savefig(VISUALIZATIONS_PATH / f'boxplot_{classifier}.pdf', bbox_inches='tight')
    plt.close()
