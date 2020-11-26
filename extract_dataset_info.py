import datasets
import numpy as np
import pandas as pd

from collections import Counter
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def extract(k=5, verbose=True):
    rows = []
    columns = ['Name', 'DI', 'IR', 'Samples', 'Features']

    for name in tqdm(datasets.names()):
        dataset = datasets.load(name)

        (X_train, y_train), (X_test, y_test) = dataset[0][0], dataset[0][1]

        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])

        n_samples = X.shape[0]
        n_features = X.shape[1]

        majority_class = Counter(y).most_common()[0][0]

        n_majority_samples = Counter(y).most_common()[0][1]
        n_minority_samples = Counter(y).most_common()[1][1]

        imbalance_ratio = np.round(n_majority_samples / n_minority_samples, 2)

        knn = NearestNeighbors(k + 1).fit(X)

        difficulty_coefficients = []

        for X_i, y_i in zip(X, y):
            if y_i == majority_class:
                continue
            else:
                indices = knn.kneighbors([X_i], return_distance=False)[0, 1:]
                n_majority_neighbors = sum(y[indices] == majority_class)

                difficulty_coefficients.append(n_majority_neighbors / k)

        difficulty_index = np.round(np.mean(difficulty_coefficients), 3)

        rows.append([name, difficulty_index, imbalance_ratio, n_samples, n_features])

    df = pd.DataFrame(rows, columns=columns)
    df = df.sort_values('DI')

    df.to_csv(Path(__file__).parent / 'results' / 'dataset_info.csv', index=False)

    if verbose:
        for column in ['DI', 'IR']:
            df[column] = df[column].map(lambda x: f'{x:.2f}')

        for i in range(30):
            row = [str(df.iloc[i][c]) for c in columns]

            if i + 30 < len(df):
                row += [str(df.iloc[i + 30][c]) for c in columns]
            else:
                row += ['' for _ in columns]

            print(' & '.join(row).replace('_', '\\_') + ' \\\\')

    return df


if __name__ == '__main__':
    extract(verbose=True)
