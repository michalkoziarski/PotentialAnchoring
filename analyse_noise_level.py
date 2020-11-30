import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path


RESULTS_PATH = Path(__file__).parent / 'results'
VISUALIZATIONS_PATH = Path(__file__).parent / 'visualizations'


def visualize():
    csv_path = RESULTS_PATH / 'results_noise_level.csv'

    df = pd.read_csv(csv_path)

    df = df.groupby(
        ['Dataset', 'Classifier', 'Resampler', 'Level', 'Metric']
    )['Score'].agg('mean').reset_index()

    df['Resampler'] = df['Resampler'].replace({
        'polynom-fit-SMOTE': 'pf-SMOTE',
        'Assembled-SMOTE': 'A-SMOTE',
        'SMOTE-TomekLinks': 'SMOTE-TL'
    })

    order = [
        'SMOTE', 'pf-SMOTE', 'Lee', 'PA',
        'SMOBD', 'G-SMOTE', 'LVQ-SMOTE',
        'A-SMOTE', 'SMOTE-TL', 'RBO'
    ]

    g = sns.FacetGrid(
        df, col='Classifier', row='Metric', sharey=False,
        row_order=['AUC', 'G-mean'],
        hue='Resampler', palette='muted', hue_order=order,
        hue_kws={
            'alpha': [1.0 if o_i == 'PA' else 0.15 for o_i in order],
            'linewidth': [3.0 if o_i == 'PA' else 1.0 for o_i in order],
            'marker': ['o'] * len(order)
        }
    )
    g.map(
        sns.lineplot, 'Level', 'Score', ci=None
    )
    g.set_titles('{row_name}, {col_name}')
    g.set_xticklabels(rotation=45)

    for ax in g.axes.flat:
        ax.tick_params(axis='y', labelleft=True)
        ax.set_xlim((-0.02, 0.22))
        ax.set_xlim((-0.02, 0.22))

    VISUALIZATIONS_PATH.mkdir(exist_ok=True, parents=True)

    plt.savefig(VISUALIZATIONS_PATH / f'noise_level.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    visualize()
