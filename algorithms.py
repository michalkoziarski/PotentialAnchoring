import numpy as np
import torch

from sklearn.cluster import KMeans
from tqdm import tqdm


def rbf(d, gamma):
    return torch.exp(-d.div(gamma).pow(2))


def potential(x, points, gamma):
    result = 0.0

    for point in points:
        result += rbf(torch.dist(x, point), gamma)

    return result


def normalize(v):
    n = torch.norm(v, p=2).detach()

    return v.div(n)


def loss_function(anchors, prototypes, points, gamma):
    real_potentials = torch.zeros(anchors.shape[0])
    prototype_potentials = torch.zeros(anchors.shape[0])

    for i, anchor in enumerate(anchors):
        real_potentials[i] = potential(anchor, points, gamma)
        prototype_potentials[i] = potential(anchor, prototypes, gamma)

    return ((normalize(real_potentials) - normalize(prototype_potentials)) ** 2).mean()


class PAO:
    def __init__(self, gamma=0.25, n_anchors=25, learning_rate=10.0, iterations=500,
                 tolerance=1e-6, epsilon=1e-4, minority_class=None, n=None, seed=None,
                 device=torch.device('cpu')):
        self.gamma = gamma
        self.n_anchors = n_anchors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.minority_class = minority_class
        self.n = n
        self.seed = seed
        self.device = device

        self._anchors = None
        self._prototypes = None
        self._loss = None

    def fit_sample(self, X, y):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.random.manual_seed(self.seed)

        classes = np.unique(y)

        assert len(classes) == 2

        if self.minority_class is None:
            sizes = [sum(y == c) for c in classes]

            minority_class = classes[np.argmin(sizes)]
            majority_class = classes[np.argmax(sizes)]
        else:
            minority_class = self.minority_class

            if classes[0] != minority_class:
                majority_class = classes[0]
            else:
                majority_class = classes[1]

        minority_points = X[y == minority_class]
        majority_points = X[y == majority_class]

        if self.n is None:
            n = len(majority_points) - len(minority_points)
        else:
            n = self.n

        if n == 0:
            return X, y

        self._anchors = torch.tensor(
            KMeans(n_clusters=self.n_anchors, random_state=self.seed).fit(X).cluster_centers_,
            device=self.device, requires_grad=False, dtype=torch.float
        )

        self._prototypes = torch.tensor(
            minority_points[np.random.randint(minority_points.shape[0], size=n), :] +
            np.random.normal(scale=self.epsilon, size=(n, minority_points.shape[1])),
            device=self.device, requires_grad=True, dtype=torch.float
        )

        minority_points = torch.tensor(
            minority_points,
            device=self.device, requires_grad=False, dtype=torch.float
        )

        optimizer = torch.optim.SGD([self._prototypes], lr=self.learning_rate)

        self._loss = []

        with tqdm(total=self.iterations) as pbar:
            for i in range(self.iterations):
                loss = loss_function(self._anchors, self._prototypes, minority_points, self.gamma)

                if self.tolerance is not None and len(self._loss) > 1 and self._loss[-2] - self._loss[-1] < self.tolerance:
                    break

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                self._loss.append(loss.data.item())

                pbar.set_description(f'Iteration {i + 1}: loss = {self._loss[-1]:.5f}')
                pbar.update()

        self._anchors = self._anchors.cpu().detach().numpy()
        self._prototypes = self._prototypes.cpu().detach().numpy()

        return np.concatenate([X, self._prototypes]), np.concatenate([y, minority_class * np.ones(n)])
