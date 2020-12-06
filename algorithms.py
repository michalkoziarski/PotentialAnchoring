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


def regularization_term(prototypes, starting_positions, gamma):
    result = 0.0

    for prototype, starting_position in zip(prototypes, starting_positions):
        result += rbf(torch.dist(prototype, starting_position), gamma)

    return result


def loss_function(anchors, prototypes, starting_positions, reference_potential, gamma, lambd=0.0):
    loss = ((reference_potential - normalized_potential(anchors, prototypes, gamma)) ** 2).mean()

    if lambd > 0.0:
        loss += lambd * regularization_term(prototypes, starting_positions, gamma)

    return loss


class AbstractPA:
    def __init__(self, kind, gamma=0.5, lambd=0.0, n_anchors=10, learning_rate=0.0001,
                 iterations=200, epsilon=1e-4, minority_class=None, n=None, ratio=None,
                 random_state=None, device=torch.device('cpu')):
        assert kind in ['oversample', 'undersample']

        self.kind = kind
        self.gamma = gamma
        self.lambd = lambd
        self.n_anchors = n_anchors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon
        self.minority_class = minority_class
        self.n = n
        self.ratio = ratio
        self.random_state = random_state
        self.device = device

        self._anchors = None
        self._prototypes = None
        self._loss = None

    def sample(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.random.manual_seed(self.random_state)

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

        assert not ((self.n is not None) and (self.ratio is not None))

        if self.n is not None:
            n = self.n
        elif self.ratio is not None:
            if self.kind == 'oversample':
                n = int(self.ratio * (len(majority_points) - len(minority_points)))
            else:
                n = len(majority_points) - int(self.ratio * (len(majority_points) - len(minority_points)))
        else:
            if self.kind == 'oversample':
                n = len(majority_points) - len(minority_points)
            else:
                n = len(minority_points)

        if (self.kind == 'oversample' and n == 0) or (self.kind != 'oversample' and n == len(majority_points)):
            return X, y

        self._anchors = torch.tensor(
            KMeans(n_clusters=self.n_anchors, random_state=self.random_state).fit(X).cluster_centers_,
            device=self.device, requires_grad=False, dtype=torch.float
        )

        if self.kind == 'oversample':
            reference_points = minority_points
        else:
            reference_points = majority_points

        indices = np.random.randint(reference_points.shape[0], size=n)

        self._prototypes = torch.tensor(
            reference_points[indices, :] + np.random.normal(scale=self.epsilon, size=(n, reference_points.shape[1])),
            device=self.device, requires_grad=True, dtype=torch.float
        )

        starting_positions = torch.tensor(
            reference_points[indices, :], device=self.device, requires_grad=False, dtype=torch.float
        )

        reference_points = torch.tensor(reference_points, device=self.device, requires_grad=False, dtype=torch.float)
        reference_potential = normalized_potential(self._anchors, reference_points, self.gamma)

        optimizer = torch.optim.Adam([self._prototypes], lr=self.learning_rate)

        self._loss = []

        with tqdm(total=self.iterations) as pbar:
            for i in range(self.iterations):
                loss = loss_function(
                    self._anchors, self._prototypes, starting_positions,
                    reference_potential, self.gamma, self.lambd
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                self._loss.append(loss.data.item())

                if np.isnan(self._loss[-1]):
                    raise RuntimeError('Convergence error (loss is NaN).')

                pbar.set_description(f'Iteration {i + 1}: loss = {self._loss[-1]:.7f}')
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
    def __init__(self, gamma=0.5, lambd=0.0, n_anchors=10, learning_rate=0.0001,
                 iterations=200, epsilon=1e-4, minority_class=None, n=None, ratio=None,
                 random_state=None, device=torch.device('cpu')):
        super().__init__(
            kind='oversample', gamma=gamma, lambd=lambd, n_anchors=n_anchors,
            learning_rate=learning_rate, iterations=iterations,
            epsilon=epsilon, minority_class=minority_class,
            n=n, ratio=ratio, random_state=random_state, device=device
        )


class PAU(AbstractPA):
    def __init__(self, gamma=0.5, lambd=0.0, n_anchors=10, learning_rate=0.0001,
                 iterations=200, epsilon=1e-4, minority_class=None, n=None, ratio=None,
                 random_state=None, device=torch.device('cpu')):
        super().__init__(
            kind='undersample', gamma=gamma, lambd=lambd, n_anchors=n_anchors,
            learning_rate=learning_rate, iterations=iterations,
            epsilon=epsilon, minority_class=minority_class,
            n=n, ratio=ratio, random_state=random_state, device=device
        )


class PA:
    def __init__(self, ratio=0.1, gamma=0.5, lambd=0.0, n_anchors=10, learning_rate=0.0001,
                 iterations=200, epsilon=1e-4, minority_class=None, random_state=None,
                 device=torch.device('cpu')):
        assert 0 <= ratio <= 1

        self.ratio = ratio
        self.gamma = gamma
        self.lambd = lambd
        self.n_anchors = n_anchors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon
        self.minority_class = minority_class
        self.random_state = random_state
        self.device = device

        self.pao = PAO(
            gamma=gamma, lambd=lambd, n_anchors=n_anchors, learning_rate=learning_rate,
            iterations=iterations, epsilon=epsilon, minority_class=minority_class,
            random_state=random_state, device=device
        )

        self.pau = PAU(
            gamma=gamma, lambd=lambd, n_anchors=n_anchors, learning_rate=learning_rate,
            iterations=iterations, epsilon=epsilon, minority_class=minority_class,
            random_state=random_state, device=device
        )

    def sample(self, X, y):
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

        n = len(majority_points) - len(minority_points)

        self.pao.n = int(self.ratio * n)
        self.pau.n = len(majority_points) - int((1 - self.ratio) * n)

        self.pao.sample(X, y)
        self.pau.sample(X, y)

        if self.pau._prototypes is None:
            X_, y_ = X, y
        else:
            X_ = np.concatenate([minority_points, self.pau._prototypes])
            y_ = np.concatenate([
                minority_class * np.ones(len(minority_points)),
                majority_class * np.ones(len(self.pau._prototypes))
            ])

        if self.pao._prototypes is not None:
            X_ = np.concatenate([X_, self.pao._prototypes])
            y_ = np.concatenate([y_, minority_class * np.ones(len(self.pao._prototypes))])

        return X_, y_
