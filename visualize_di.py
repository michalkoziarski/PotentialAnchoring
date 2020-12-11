from sklearn.neighbors import NearestNeighbors
from visualization import *


def difficulty_index(X, y, k=5, majority_class=0):
    knn = NearestNeighbors(k + 1).fit(X)

    difficulty_coefficients = []

    for X_i, y_i in zip(X, y):
        if y_i == majority_class:
            continue
        else:
            indices = knn.kneighbors([X_i], return_distance=False)[0, 1:]
            n_majority_neighbors = sum(y[indices] == majority_class)

            difficulty_coefficients.append(n_majority_neighbors / k)

    return np.round(np.mean(difficulty_coefficients), 3)


def create_dataset(r, n, k):
    np.random.seed(42)

    X = np.random.uniform(size=(1000, 2))
    y = np.array([1 if distance(X_i, (0.5, 0.5)) < r else 0 for X_i in X])

    flipped_indices = np.random.choice(np.where(y == 0)[0], n, replace=False)
    y[flipped_indices] = 1

    if k > 0:
        knn = NearestNeighbors(k).fit(X)

        for i in flipped_indices:
            neighbor_indices = knn.kneighbors([X[i]], return_distance=False)[0, 1:]
            y[neighbor_indices] = 1

    return X, y


if __name__ == '__main__':
    params = [
        {
            'r': 0.2,
            'n': 0,
            'k': 0
        },
        {
            'r': 0.15,
            'n': 5,
            'k': 25
        },
        {
            'r': 0.1,
            'n': 25,
            'k': 3
        },
        {
            'r': 0.0,
            'n': 100,
            'k': 0
        }
    ]

    for i, p in enumerate(params):
        X, y = create_dataset(**p)
        di = difficulty_index(X, y)

        visualize(
            X, y,
            file_name=f'example_di_{i}_{di}',
            minority_class=1
        )
