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
    return v.div(torch.norm(v, p=2).detach())


def normalized_potential(anchors, points, gamma):
    result = torch.zeros(anchors.shape[0])

    for i, anchor in enumerate(anchors):
        result[i] = potential(anchor, points, gamma)

    result = normalize(result)

    return result


def loss_function(anchors, prototypes, gamma, reference_potential):
    return ((reference_potential - normalized_potential(anchors, prototypes, gamma)) ** 2).mean()


class AbstractPA:
    def __init__(self, kind, gamma=0.25, n_anchors=25, learning_rate=0.001, iterations=200,
                 tolerance=1e-6, epsilon=1e-4, minority_class=None, n=None, seed=None,
                 device=torch.device('cpu')):
        assert kind in ['oversample', 'undersample']

        self.kind = kind
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

        if self.kind == 'oversample':
            reference_points = minority_points
        else:
            reference_points = majority_points

        self._prototypes = torch.tensor(
            reference_points[np.random.randint(reference_points.shape[0], size=n), :] +
            np.random.normal(scale=self.epsilon, size=(n, reference_points.shape[1])),
            device=self.device, requires_grad=True, dtype=torch.float
        )

        reference_points = torch.tensor(reference_points, device=self.device, requires_grad=False, dtype=torch.float)
        reference_potential = normalized_potential(self._anchors, reference_points, self.gamma)

        optimizer = torch.optim.Adam([self._prototypes], lr=self.learning_rate)

        self._loss = []

        with tqdm(total=self.iterations) as pbar:
            for i in range(self.iterations):
                loss = loss_function(self._anchors, self._prototypes, self.gamma, reference_potential)

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

        if self.kind == 'oversample':
            X_ = np.concatenate([X, self._prototypes])
            y_ = np.concatenate([y, minority_class * np.ones(n)])
        else:
            X_ = np.concatenate([minority_points, self._prototypes])
            y_ = np.concatenate([minority_class * np.ones(len(minority_points)), majority_class * np.ones(n)])

        return X_, y_


class PAO(AbstractPA):
    def __init__(self, gamma=0.25, n_anchors=25, learning_rate=0.001, iterations=200,
                 tolerance=1e-6, epsilon=1e-4, minority_class=None, n=None, seed=None,
                 device=torch.device('cpu')):
        super().__init__(
            kind='oversample', gamma=gamma, n_anchors=n_anchors,
            learning_rate=learning_rate, iterations=iterations,
            tolerance=tolerance, epsilon=epsilon,
            minority_class=minority_class,
            n=n, seed=seed, device=device
        )


class PAU(AbstractPA):
    def __init__(self, gamma=0.25, n_anchors=25, learning_rate=0.001, iterations=200,
                 tolerance=1e-6, epsilon=1e-4, minority_class=None, n=None, seed=None,
                 device=torch.device('cpu')):
        super().__init__(
            kind='undersample', gamma=gamma, n_anchors=n_anchors,
            learning_rate=learning_rate, iterations=iterations,
            tolerance=tolerance, epsilon=epsilon,
            minority_class=minority_class,
            n=n, seed=seed, device=device
        )
