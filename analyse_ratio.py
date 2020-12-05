import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path


VISUALIZATIONS_PATH = Path(__file__).parent / 'visualizations'


if __name__ == '__main__':
    VISUALIZATIONS_PATH.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(f'results/results_ratio.csv')

    g = sns.catplot(
        data=df, x='Ratio', y='Score', row='Classifier', col='Metric',
        sharey='col', height=2.0, aspect=1.4, kind='point', ci=None,
        hue='Metric', palette='muted'
    )
    g.set_titles('{row_name}, {col_name}')
    g.set_xticklabels(rotation=45)

    for ax in g.axes.flat:
        ax.tick_params(axis='y', labelleft=True)

    plt.savefig(VISUALIZATIONS_PATH / f'ratio.pdf', bbox_inches='tight')
