import numpy as np

from collections import Counter


def distance(x, y, p_norm=2):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def rbf(d, gamma):
    if gamma == 0.0:
        return 0.0
    else:
        return np.exp(-(d / gamma) ** 2)


def mutual_class_potential(point, majority_points, minority_points, gamma, p_norm):
    result = 0.0

    for majority_point in majority_points:
        result += rbf(distance(point, majority_point, p_norm), gamma)

    for minority_point in minority_points:
        result -= rbf(distance(point, minority_point, p_norm), gamma)

    return result


class RBU:
    def __init__(self, gamma=0.05, ratio=1.0, p_norm=2, minority_class=None):
        self.gamma = gamma
        self.ratio = ratio
        self.p_norm = p_norm
        self.minority_class = minority_class

    def sample(self, X, y):
        X = X.copy()
        y = y.copy()

        if self.minority_class is None:
            minority_class = Counter(y).most_common()[1][0]
            majority_class = Counter(y).most_common()[0][0]
        else:
            classes = np.unique(y)

            minority_class = self.minority_class

            if classes[0] != minority_class:
                majority_class = classes[0]
            else:
                majority_class = classes[1]

        minority_points = X[y == minority_class]
        majority_points = X[y == majority_class]

        n = int(np.round(self.ratio * (len(majority_points) - len(minority_points))))

        majority_potentials = np.zeros(len(majority_points))

        for i, point in enumerate(majority_points):
            majority_potentials[i] = mutual_class_potential(
                point, majority_points, minority_points, self.gamma, self.p_norm
            )

        deleted_majority_points = []

        for _ in range(n):
            idx = np.argmax(majority_potentials)

            for i, point in enumerate(majority_points):
                majority_potentials[i] -= rbf(distance(point, majority_points[idx], self.p_norm), self.gamma)

            deleted_majority_points.append(majority_points[idx])

            majority_points = np.delete(majority_points, idx, axis=0)
            majority_potentials = np.delete(majority_potentials, idx)

        X = np.concatenate([majority_points, minority_points])
        y = np.concatenate([
            np.tile(majority_class, len(majority_points)),
            np.tile(minority_class, len(minority_points))
        ])

        return X, y
